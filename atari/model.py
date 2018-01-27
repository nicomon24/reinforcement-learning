'''
    Definition of the DQN architecture
'''

import tensorflow as tf

class DQN:

    def __init__(self, input_size, action_space_size, scope, reuse=False):
        # Input size of type (height, width, time), e.g. (83, 83, 4)
        self.input_size = input_size
        self.action_space_size = action_space_size
        with tf.variable_scope(scope, reuse=reuse) as my_scope:
            # Copy scope
            self.scope = my_scope
            # Define input
            self.input = tf.placeholder(tf.float32, shape=(None, input_size[0], input_size[1], input_size[2]))
            # Transpose to place the time on the last channel
            transposed = tf.transpose(self.input, [0, 2, 3, 1])
            # initializer
            initializer = tf.contrib.layers.variance_scaling_initializer()
            # First convolution, 32 8x8 filters, stride=4
            conv1 = tf.layers.conv2d(transposed, filters=32, kernel_size=(8, 8), padding='SAME', strides=4, activation=tf.nn.relu, kernel_initializer=initializer)
            # Second convolution, 64 4x4, stride=2
            conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=(4, 4), padding='SAME', strides=2, activation=tf.nn.relu, kernel_initializer=initializer)
            # Third convolution, 64 3x3, stride=1
            conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=(3, 3), padding='SAME', strides=1, activation=tf.nn.relu, kernel_initializer=initializer)
            # First hidden layer
            flatted = tf.layers.flatten(conv3)
            hidden1 = tf.layers.dense(flatted, 512, activation=tf.nn.relu, kernel_initializer=initializer)
            self.logits = tf.layers.dense(hidden1, action_space_size, kernel_initializer=initializer)
            # Train op
            self.actions = tf.placeholder(tf.uint8, shape=(None,))
            self.targets = tf.placeholder(tf.float32, shape=(None,))
            logits_selected = tf.reduce_sum(tf.one_hot(self.actions, action_space_size) * self.logits, axis=-1)
            # Huber loss
            self.cost = tf.losses.huber_loss(self.targets, logits_selected)
            # Train op
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.cost)
            # Add summaries
            tf.summary.scalar('cost', self.cost)

    def copy_from(self, master_dqn, session):
        # Get all trainable vars from the 2 dqn and order them by name to have the same order
        my_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
        master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=master_dqn.scope.name)
        # Assign values
        ops = [v.assign(v_master) for v, v_master in zip(my_vars, master_vars)]
        session.run(ops)

    # Train the network given a batch of N samples of type: ()
    def train(self, batch_states, batch_actions, batch_values, session):
        _, cost = session.run([self.train_op, self.cost], feed_dict={
            self.input: batch_states,
            self.actions: batch_actions,
            self.targets: batch_values
        })
        return cost

    # Predict a batch of N samples of type: state
    def predict(self, batch, session):
        return session.run(self.logits, feed_dict={self.input: batch})
