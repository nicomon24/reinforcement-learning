'''
    This script implements the DQN algorithm as described in the DeepMind paper
    and applies it to the Atari Breakout environment.

    Can be used both for training and evaluation
'''

import tensorflow as tf
import numpy as np
import random, argparse, os
from scipy.misc import imresize
import gym
from tqdm import trange
import matplotlib.pyplot as plt
from time import sleep
import time

def encode_frame(new_frame):
    global last_frame
    # First we need to take the max from new frame and last frame to remove flickering
    # Breakout seems not to have this glitches, so I will skip it for now
    #stacked = np.stack([last_frame, new_frame], axis=-1)
    #new_frame = stacked.max(axis=-1)
    #last_frame = new_frame
    # Transform to B/W for simplicity
    bw = new_frame.mean(axis=2)
    # Crop removing useless parts
    cropped = bw[30:195,4:-4]
    # Resize to wanted size (83x83)
    resized = imresize(cropped, size=(TARGET_SIZE, TARGET_SIZE), interp='nearest')
    return resized

'''
    The replay buffer stores an amount N of previous sequences
    (s, a, r, s', done) that will be sampled for training
'''
class ReplayBuffer:

    def __init__(self, max_len):
        self.buffer = []
        self.max_len = max_len

    def add_to_buffer(self, sample):
        if len(self.buffer) >= self.max_len:
            self.buffer.pop(0) # Remove 1 sample
        # Add the new sample
        self.buffer.append(sample)

    def get_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

class DQN:

    def __init__(self, frame_size, frames_per_state, action_space_size, scope):
        self.scope = scope
        with tf.variable_scope(scope):
            self.global_step = tf.train.get_or_create_global_step()
            self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)
            # Define the network
            self.input = tf.placeholder(tf.float32, shape=(None, frames_per_state, frame_size, frame_size))
            # Transpose to place the time on the last channel
            transposed = tf.transpose(self.input, [0, 2, 3, 1])
            # First convolution, 32 8x8 filters, stride=4
            conv1 = tf.layers.conv2d(transposed, filters=32, kernel_size=(8, 8), padding='SAME', strides=4)
            relu1 = tf.nn.relu(conv1)
            # Second convolution, 64 4x4, stride=2
            conv2 = tf.layers.conv2d(relu1, filters=64, kernel_size=(4, 4), padding='SAME', strides=2)
            relu2 = tf.nn.relu(conv2)
            # Third convolution, 64 3x3, stride=1
            conv3 = tf.layers.conv2d(relu2, filters=64, kernel_size=(3, 3), padding='SAME', strides=1)
            relu3 = tf.nn.relu(conv3)
            # First hidden layer
            flatted = tf.layers.flatten(relu3)
            hidden1 = tf.layers.dense(flatted, 512)
            self.logits = tf.layers.dense(hidden1, action_space_size)
            # Train op
            self.actions = tf.placeholder(tf.uint8, shape=(None,))
            self.targets = tf.placeholder(tf.float32, shape=(None,))
            logits_selected = tf.reduce_sum(tf.one_hot(self.actions, action_space_size) * self.logits, axis=-1)
            self.cost = tf.reduce_mean(tf.square(self.targets - logits_selected))
            self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.cost)
            # Add summaries
            tf.summary.scalar('cost', self.cost)

    def copy_from(self, master_dqn, session):
        # Get all trainable vars from the 2 dqn and order them by name to have the same order
        my_vars = sorted([v for v in tf.trainable_variables() if v.name.startswith(self.scope)], key=lambda v: v.name)
        master_vars = sorted([v for v in tf.trainable_variables() if v.name.startswith(master_dqn.scope)], key=lambda v: v.name)
        # Assign values
        ops = [v.assign(session.run(v_master)) for v, v_master in zip(my_vars, master_vars)]
        session.run(ops)

def print_state(state):
    fig = plt.figure()
    for i in range(4):
        ax = fig.add_subplot(2,2,i+1)
        ax.imshow(state[i])
    plt.show()

def train(env=None, n_frames_in_state=4, action_repeat=1, dqn=None, target_dqn=None,
            saver=None, FLAGS=None, start_step = 0):
    REPLAY_BUFFER_MIN_SIZE = 50000
    REPLAY_BUFFER_MAX_SIZE = 1000000
    EPSILON_MIN = 0.1
    EPSILON_MAX = 1.0
    EPSILON_FALL = 1000000
    GAMMA = 0.99
    TRAIN_EPISODES = FLAGS.episodes
    BATCH_SIZE = 32
    COPY_STEP = 10000
    STEPS_TO_CHECKPOINT = 5000

    # Create replay buffer
    RB = ReplayBuffer(REPLAY_BUFFER_MAX_SIZE)

    # Fill replay buffer
    obs = env.reset()
    _encoded_obs = encode_frame(obs)
    state = np.array([_encoded_obs] * n_frames_in_state)
    current_game_step = 0
    done = False
    # Populate the replay buffer with the minimum number of experiences (random)
    for experience in trange(REPLAY_BUFFER_MIN_SIZE):
        # Choose a random action
        if current_game_step % action_repeat == 0:
            a = env.action_space.sample()
        # Play
        for _i in range(action_repeat):
            obs, r, done, _ = env.step(a)
        # Get the new state and add to replay buffer
        _encoded_obs = encode_frame(obs)
        next_state = np.append(state[1:], [_encoded_obs], axis=0)
        RB.add_to_buffer((state, a, r, next_state, done))
        # Check if terminated
        if done:
            obs = env.reset()
            _encoded_obs = encode_frame(obs)
            state = np.array([_encoded_obs] * n_frames_in_state)
            current_game_step = 0
        else:
            # Update game step
            current_game_step += 1
    print("Replay buffer size:", len(RB.buffer))

    total_steps = start_step
    rewards = []

    for e in range(TRAIN_EPISODES):

        obs = env.reset()
        _encoded_obs = encode_frame(obs)
        state = np.array([_encoded_obs] * n_frames_in_state)
        episode_steps = 0
        done = False
        episode_reward = 0
        a = 0

        # No-op between 0-30
        _n = np.random.randint(0,30)
        for i in range(_n):
            obs, r, done, _ = env.step(0)

        while not done:
            # Check if we need to save checkpoint
            if total_steps > start_step and total_steps % STEPS_TO_CHECKPOINT == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.alias + '.ckpt')
                tf.logging.info('Saving to "%s-%d"', checkpoint_path, total_steps)
                saver.save(sess, checkpoint_path, global_step=total_steps, write_meta_graph=False)
            # Check if we need to copy target network
            if total_steps % COPY_STEP == 0:
                target_dqn.copy_from(dqn, sess)

            # Adjust epsilon
            if total_steps < EPSILON_FALL:
                epsilon = EPSILON_MAX - (EPSILON_MAX - EPSILON_MIN) * total_steps / EPSILON_FALL
            else:
                epsilon = EPSILON_MIN
            # Epsilon greedy
            if np.random.random() < epsilon:
                # Choose random action
                a = env.action_space.sample()
            else:
                # Choose best action
                est = sess.run(dqn.logits, feed_dict={dqn.input: [state]})[0]
                a = np.argmax(est)
            for _i in range(action_repeat):
                # Perform action
                obs, r, done, _ = env.step(a)
                episode_reward += r

            # Get the new state and add to replay buffer
            _encoded_obs = encode_frame(obs)
            next_state = np.append(state[1:], [_encoded_obs], axis=0)
            RB.add_to_buffer((state, a, r, next_state, done))

            # Extract batch of states and train network
            batch = RB.get_batch(BATCH_SIZE)
            batch_states = [s[0] for s in batch]
            actions = [s[1] for s in batch]
            pred_nextQ = sess.run(target_dqn.logits, feed_dict={target_dqn.input: [s[3] for s in batch]})
            nextQ = [s[2] + GAMMA * max(y) if s[4]==False else s[2] for s,y in zip(batch, pred_nextQ)]
            # Train network
            _, cost, _step = sess.run([dqn.train_op, dqn.cost, dqn.increment_global_step], feed_dict={dqn.input: batch_states, dqn.targets:nextQ, dqn.actions: actions})
            episode_steps += 1
            total_steps += 1
        rewards.append(episode_reward)
        # Episode is done, print summary
        print("Episode:", e,
          "Num steps:", episode_steps,
          "Reward:", episode_reward,
          "Avg Reward (Last 100):", "%.3f" % np.mean(rewards[-100:]),
          "Epsilon:", "%.3f" % epsilon
        )

    # When ended, save checkpoint
    checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.alias + '.ckpt')
    tf.logging.info('Saving to "%s-%d"', checkpoint_path, TRAIN_STEPS)
    saver.save(sess, checkpoint_path, global_step=TRAIN_STEPS)

def test(env=None, n_frames_in_state=4, action_repeat=1, dqn=None):
    EPSILON = 0.05
    # Play one episode informative
    obs = env.reset()
    _encoded_obs = encode_frame(obs)
    state = np.array([_encoded_obs] * n_frames_in_state)
    current_game_step = 0
    done = False
    while not done:
        if current_game_step % action_repeat == 0:
            if np.random.random() < EPSILON:
                a = env.action_space.sample()
            else:
                est = sess.run(dqn.logits, feed_dict={dqn.input: [state]})[0]
                a = np.argmax(est)
        obs, r, done, _ = env.step(a)
        env.render()
        current_game_step += 1
    print("Ended.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
      help="""\
      Mode: train or test.
    """)
    parser.add_argument('--alias', type=str, default='tmp',
      help="""\
      Alias to create the checkpoint name
    """)
    parser.add_argument('--train_dir', type=str, default='tmp',
      help="""\
      Directory to save checkpoints
    """)
    parser.add_argument('--checkpoint', type=str, default='',
      help="""\
      Select a checkpoint to start from
    """)
    parser.add_argument('--episodes', type=int, default=100000,
      help="""\
      Number of training steps
    """)
    parser.add_argument('--action_repeat', type=int, default=4,
      help="""\
      Number of training steps
    """)
    FLAGS, unparsed = parser.parse_known_args()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Parameters
    TARGET_SIZE = 83 # Size in width and height of the image we want
    ACTION_REPEAT = FLAGS.action_repeat # Repeat the same action for N frames
    N_FRAMES_IN_STATE = 4
    GENERATOR_SCOPE = 'generator'
    TARGET_SCOPE = 'target'

    env = gym.make("Breakout-v0")
    # Set seed for replication
    env.seed(42)

    sess = tf.Session()
    # Init networks
    dqn = DQN(frame_size=TARGET_SIZE, frames_per_state=N_FRAMES_IN_STATE,
                    action_space_size=env.action_space.n, scope=GENERATOR_SCOPE)
    target_dqn = DQN(frame_size=TARGET_SIZE, frames_per_state=N_FRAMES_IN_STATE,
                    action_space_size=env.action_space.n, scope=TARGET_SCOPE)
    # Init tf variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=GENERATOR_SCOPE), max_to_keep=10)

    start_step = 0
    # Check if we need to load a checkpoint
    if FLAGS.checkpoint:
        _saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=GENERATOR_SCOPE))
        _saver.restore(sess, FLAGS.checkpoint)
        start_step = dqn.global_step.eval(session=sess)

    if FLAGS.mode == 'train':
        train(env=env, n_frames_in_state=N_FRAMES_IN_STATE, action_repeat=ACTION_REPEAT,
                dqn=dqn, target_dqn=target_dqn, saver=saver, FLAGS=FLAGS, start_step=start_step)
    elif FLAGS.mode == 'test':
        test(env=env, n_frames_in_state=N_FRAMES_IN_STATE, action_repeat=ACTION_REPEAT, dqn=dqn)
