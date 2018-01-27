'''
    This script implements the DQN algorithm as described in the DeepMind paper
    and applies it to the Atari Breakout environment.

    Can be used both for training and evaluation

    ACTIONS:
    - 0: noop
    - 1: fire
    - 2: right
    - 3: left
'''

import tensorflow as tf
from tqdm import trange
import numpy as np
import random, argparse, os
import gym
from time import sleep
import time
from collections import deque

from wrappers import SkipperEnvironment, CroppedEnvironment, TimeStackEnvironment, FiringEnvironment, NoCatsLivesEnvironment, ClipRewardEnv, NoopEnvironment
from model import DQN
from replay_buffer import ReplayBuffer

'''
import matplotlib.pyplot as plt

def plot_state(state):
    fig = plt.figure()
    axes = [plt.subplot(221), plt.subplot(222), plt.subplot(223), plt.subplot(224)]
    for i, axis in enumerate(axes):
        axis.imshow(state[:,:,i])
    plt.show()
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
      help="""\
      Mode: train | test | render
    """)
    parser.add_argument('--alias', type=str, default='tmp',
      help="""\
      Alias to create the checkpoint name.
    """)
    parser.add_argument('--train_dir', type=str, default='tmp',
      help="""\
      Directory to save checkpoints.
    """)
    parser.add_argument('--log_dir', type=str, default='tmp_logs',
      help="""\
      Directory to save logs for tensorboard.
    """)
    parser.add_argument('--checkpoint', type=str, default='',
      help="""\
      Select a checkpoint to start from.
    """)
    parser.add_argument('--video_output', type=str, default='',
      help="""\
      Select the filename of the output video (only in render mode).
    """)
    parser.add_argument('--render_episodes', type=int, default=10,
      help="""\
      Select the number of episodes to render.
    """)
    parser.add_argument('--steps', type=int, default=30000000,
      help="""\
      Number of training steps.
    """)
    parser.add_argument('--start_steps', type=int, default=0,
      help="""\
      Number of initial training steps.
    """)
    parser.add_argument('--vanilla', type=int, default=0,
      help="""\
      Environment reward as designed or modified.
    """)
    FLAGS, unparsed = parser.parse_known_args()
    tf.logging.set_verbosity(tf.logging.INFO)

    FRAME_SIZE = 83
    TIME_STACK = 4
    STEPS_TO_TRAIN = 4
    REPLAY_BUFFER_MIN_SIZE = 50000
    REPLAY_BUFFER_MAX_SIZE = 1000000
    EPSILON_MIN = 0.1
    EPSILON_MAX = 1.0
    EPSILON_FALL = 1000000
    GAMMA = 0.99
    TRAIN_STEPS = FLAGS.steps
    BATCH_SIZE = 32
    COPY_STEP = 4096
    STEPS_TO_CHECKPOINT = 10000
    GENERATOR_SCOPE = 'generator'
    TARGET_SCOPE = 'target'

    env = gym.make("BreakoutNoFrameskip-v4")
    if not FLAGS.vanilla:
        env = NoCatsLivesEnvironment(env)
    env = NoopEnvironment(env)
    env = SkipperEnvironment(env)
    env = CroppedEnvironment(env, target_size=FRAME_SIZE)
    env = TimeStackEnvironment(env, time_period=TIME_STACK)
    env = FiringEnvironment(env)
    if not FLAGS.vanilla:
        env = ClipRewardEnv(env)
    # Set seed for replication
    env.seed(42)

    # Create ReplayBuffer
    repbuf = ReplayBuffer(REPLAY_BUFFER_MAX_SIZE)

    # Create DQNs
    sess = tf.Session()
    dqn = DQN((FRAME_SIZE, FRAME_SIZE, TIME_STACK), env.action_space.n, GENERATOR_SCOPE)
    target_dqn = DQN((FRAME_SIZE, FRAME_SIZE, TIME_STACK), env.action_space.n, TARGET_SCOPE, reuse=tf.AUTO_REUSE)
    # Init vars and create saver
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=GENERATOR_SCOPE), max_to_keep=10)

    # Check if we need to load a checkpoint
    if FLAGS.checkpoint:
        _saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=GENERATOR_SCOPE))
        _saver.restore(sess, FLAGS.checkpoint)

    if FLAGS.mode == 'train':
        # Fill replay buffer with the minimum number of sample
        obs = env.reset()
        for i in trange(REPLAY_BUFFER_MIN_SIZE):
            _old_state = obs
            # Generate random play
            a = env.action_space.sample()
            obs, reward, done, _ = env.step(a)
            # Add to replay buffer, sample of type: (state, action, reward, next_state, done)
            repbuf.add_sample((_old_state, a, reward, obs, done))
            # Check if done and reset if the game is ended
            if done:
                obs = env.reset()
        print("Loaded replay buffer with", len(repbuf.buffer), "samples.")

        # Now start training
        done = True
        episode_reward = 0
        episode_steps = 0
        rewards = deque([], maxlen=1000)
        epsilon = EPSILON_MAX
        episode_rewards_100 = 0
        episode_rewards_1000 = 0
        n_episode = 0
        total_max_q = 0
        mean_max_q = 0.0

        # Create summary
        writer = tf.summary.FileWriter(FLAGS.log_dir)
        tf_epsilon = tf.placeholder(tf.float32, ())
        tf_reward = tf.placeholder(tf.float32, ())
        tf_reward100 = tf.placeholder(tf.float32, ())
        tf_reward1000 = tf.placeholder(tf.float32, ())
        tf_episode_steps = tf.placeholder(tf.int32, ())
        tf_mean_max_q = tf.placeholder(tf.float32, ())
        epsilon_summary = tf.summary.scalar('epsilon', tf_epsilon)
        reward_summary = tf.summary.scalar('reward', tf_reward)
        reward100_summary = tf.summary.scalar('reward100', tf_reward100)
        reward1000_summary = tf.summary.scalar('reward1000', tf_reward1000)
        episode_steps_summary = tf.summary.scalar('episode_steps', tf_episode_steps)
        mean_max_q_summary = tf.summary.scalar('mean_max_q', tf_mean_max_q)

        merged_summaries = tf.summary.merge([epsilon_summary, reward_summary, reward100_summary,
                            reward1000_summary, episode_steps_summary, mean_max_q_summary])

        for step in range(FLAGS.start_steps, TRAIN_STEPS):
            # Check if we need to save checkpoint
            if step > 0 and step % STEPS_TO_CHECKPOINT == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.alias + '.ckpt')
                tf.logging.info('Saving to "%s-%d"', checkpoint_path, step)
                saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=True)
            # Check if we need to copy target network
            if step % COPY_STEP == 0:
                print("Copycat")
                target_dqn.copy_from(dqn, sess)
            # Check if the environment has ended
            if done:
                # Get summary values
                if episode_steps == 0:
                    episode_steps = 1
                rewards.append(episode_reward)
                episode_rewards_100 = np.mean(list(rewards)[-100:])
                episode_rewards_1000 = np.mean(list(rewards)[-1000:])
                mean_max_q = total_max_q / episode_steps
                # Write summary
                summary = sess.run(merged_summaries, feed_dict={
                    tf_epsilon: epsilon,
                    tf_reward: episode_reward,
                    tf_reward100: episode_rewards_100,
                    tf_reward1000: episode_rewards_1000,
                    tf_episode_steps: episode_steps,
                    tf_mean_max_q: mean_max_q
                })
                writer.add_summary(summary, n_episode)
                # Print to console
                print("Step:", step,
                  "Episode:", n_episode,
                  "Num steps:", episode_steps,
                  "Reward:", episode_reward,
                  "Avg Reward (Last 100):", "%.3f" % episode_rewards_100,
                  "Avg Reward (Last 1000):", "%.3f" % episode_rewards_1000,
                  "Epsilon:", "%.3f" % epsilon,
                  "Total-max-q", total_max_q,
                  "Mean-max-q", "%.3f" % mean_max_q
                )
                # Update episode number
                n_episode += 1
                # Reset environment
                state = env.reset()
                episode_reward = 0
                episode_steps = 0
                total_max_q = 0
                done = False
            # Evaluate epsilon for epsilon-greedy policy
            if step < EPSILON_FALL:
                epsilon = EPSILON_MAX - (EPSILON_MAX - EPSILON_MIN) * step / EPSILON_FALL
            else:
                epsilon = EPSILON_MIN
            # Choose action epsilon-greedily using dqn
            q_values = dqn.predict([state], sess)[0]
            if np.random.random() < epsilon:
                # Choose random action
                a = env.action_space.sample()
            else:
                # Choose best action
                a = np.argmax(q_values)
            # Perform choosen action
            next_state, reward, done, _ = env.step(a)
            episode_reward += reward
            episode_steps += 1
            # Insert into replay buffer
            repbuf.add_sample((state, a, reward, next_state, done))
            state = next_state
            # Stats
            total_max_q += q_values.max()
            # Check if we need to train
            if step % STEPS_TO_TRAIN == 0:
                # Get a batch from replaybuffer
                batch = repbuf.get_batch(BATCH_SIZE)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
                pred_nextQ = sess.run(target_dqn.logits, feed_dict={target_dqn.input: next_state_batch})
                max_nextQ = np.max(pred_nextQ, axis=1)
                pred_values = np.array(reward_batch) + np.invert(done_batch).astype('float32') * GAMMA * max_nextQ
                cost = dqn.train(state_batch, action_batch, pred_values, sess)


    elif FLAGS.mode == 'test':
        # Testing mode
        epsilon = 0.05
        rewards = []
        for _ in trange(100):
            done = False
            obs = env.reset()
            reward = 0
            while not done:
                # Choose action
                if np.random.random() < epsilon:
                    # Choose random action
                    a = env.action_space.sample()
                else:
                    # Choose best action
                    a = np.argmax(dqn.predict([obs], sess)[0])
                obs, r, done, _ = env.step(a)
                reward += r
            rewards.append(reward)
        print("Mean 100 episode total reward:", np.mean(rewards))

    elif FLAGS.mode == 'render':
        env = gym.wrappers.Monitor(env, FLAGS.video_output)
        # Render mode
        epsilon = 0.05
        rewards = []
        for _ in trange(FLAGS.render_episodes):
            done = False
            obs = env.reset()
            reward = 0
            while not done:
                # Choose action
                if np.random.random() < epsilon:
                    # Choose random action
                    a = env.action_space.sample()
                else:
                    # Choose best action
                    a = np.argmax(dqn.predict([obs], sess)[0])
                obs, r, done, _ = env.step(a)
                reward += r
                env.render()
                #sleep(0.05)
            rewards.append(reward)
        print("Mean", FLAGS.render_episodes, "episode total reward:", np.mean(rewards))
        print("Saving to file", FLAGS.video_output)
