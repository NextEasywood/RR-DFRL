import tensorflow as tf
# import tensorflow.contrib as tc
from replay_buffer import ReplayBuffer

# import random
seed = 1
# random.seed(seed)
tf.set_random_seed(seed)


class DDPG:
    def __init__(self, name, layer_norm=True, nb_actions=3, nb_input=15, max_memory_size=20000):
        self.layer_norm = layer_norm
        self.name = name
        self.nb_actions = nb_actions
        self.nb_input = nb_input
        self.memory = ReplayBuffer(max_memory_size)

        with tf.compat.v1.variable_scope(self.name):
            self.state_input = tf.placeholder(shape=[None, self.nb_input], dtype=tf.float32, name="state_input")
            self.action_input = tf.placeholder(shape=[None, self.nb_actions], dtype=tf.float32, name="action_input")
            self.reward = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="reward")
            self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="target_Q")

            self.actor_model = self.actor_network()
            with tf.compat.v1.variable_scope('target_actor'):
                self.target_actor_model = self.actor_network()

            with tf.compat.v1.variable_scope('critic'):
                self.critic_model1 = self.critic_network(self.action_input)

            with tf.compat.v1.variable_scope('target_critic'):
                self.target_critic_model = self.critic_network(self.action_input)

            with tf.compat.v1.variable_scope('critic'):
                self.critic_model_with_actor = self.critic_network(self.actor_model, reuse=True)

            self.actor_train = tf.train.AdamOptimizer(1e-4).minimize(self.actor_loss(self.critic_model_with_actor))
            self.critic_train1 = tf.train.AdamOptimizer(1e-3).minimize(self.critic_loss(self.critic_model1))

        self.update_target_network_params()

    def actor_network(self):
        with tf.compat.v1.variable_scope('actor', reuse=tf.compat.v1.AUTO_REUSE):



            x = tf.compat.v1.layers.dense(self.state_input, 256, activation=None, kernel_initializer=tf.initializers.glorot_normal())
            x = tf.compat.v1.layers.dense(x, 256, activation=None,
                                          kernel_initializer=tf.initializers.glorot_normal())

            if self.layer_norm:
                x = tf.compat.v1.layers.batch_normalization(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.compat.v1.layers.dense(x, 256, activation=None, kernel_initializer=tf.initializers.glorot_normal())
            if self.layer_norm:
                x = tf.compat.v1.layers.batch_normalization(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.compat.v1.layers.dense(x, self.nb_actions, activation=tf.nn.tanh, kernel_initializer=tf.initializers.glorot_normal())
        return x

    def critic_network(self, action_input, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope('critic', reuse=reuse):
            x = tf.concat([self.state_input, action_input], axis=-1)
            x = tf.compat.v1.layers.dense(x, 256, activation=None, kernel_initializer=tf.initializers.glorot_normal())
            if self.layer_norm:
                x = tf.compat.v1.layers.batch_normalization(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.compat.v1.layers.dense(x, 256, activation=None, kernel_initializer=tf.initializers.glorot_normal())
            if self.layer_norm:
                x = tf.compat.v1.layers.batch_normalization(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.compat.v1.layers.dense(x, 1, kernel_initializer=tf.initializers.glorot_normal())
        return x

    def train_actor(self, state, sess):
        sess.run(self.actor_train, {self.state_input: state})

    def train_critic(self, state, action, target, sess):
        sess.run(self.critic_train1, {self.state_input: state, self.action_input: action, self.target_Q: target})

    def action(self, state, sess):
        return sess.run(self.actor_model, {self.state_input: state})

    def target_action(self, state, sess):
        return sess.run(self.target_actor_model, {self.state_input: state})

    def target_actions(self, state_batch, sess):
        return sess.run(self.target_actor_model, feed_dict={self.state_input: state_batch
        })


    def Q(self, state, action, sess, target=False):
        if target:
            return sess.run(self.target_critic_model, {self.state_input: state, self.action_input: action})
        else:
            return sess.run(self.critic_model1, {self.state_input: state, self.action_input: action})


    def actor_loss(self, critic_model1_with_actor):
        return -tf.reduce_mean(input_tensor=critic_model1_with_actor)

    def critic_loss(self, critic_model):
        return tf.compat.v1.losses.mean_squared_error(self.target_Q, critic_model)

    def get_parameters(self, scope):
        # 获取神经网络参数
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def print_parameters(self, agent_name, sess):
        parameters = sess.run(self.get_parameters(self.name + '/critic'))
        print(f'{agent_name} Critic Parameters:')
        for param in parameters:
            print(param)

    def get_critic1_params(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/critic')

    def backup_parameters(self, source_agent, sess):
        source_critic_params = source_agent.get_critic_params()
        target_critic_params = self.get_critic_params()

        copy_critic1_ops = [tf.assign(target, source) for target, source in zip(target_critic_params, source_critic_params)]

        sess.run(copy_critic1_ops)
    def copy_parameters(self, source_agent, sess):
        source_critic_params = source_agent.get_critic_params()
        target_critic_params = self.get_critic_params()

        copy_critic_ops = [tf.assign(target, (0.99*source + 0.01*target)) for target, source in zip(target_critic_params, source_critic_params)]

        sess.run(copy_critic_ops)

    def get_critic_params(self):
        # Assuming self.critic_scope is the TF scope of the critic network
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/critic')



    # 网络初始化
    def update_target_network_params(self):
        tau = 0.001  # Polyak averaging parameter
        # Assuming target networks exist and are initialized elsewhere
        actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/actor')
        target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/target_actor')
        critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/critic')
        target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/target_critic')

        update_actor = [target_actor_vars[i].assign(tau * actor_vars[i] + (1 - tau) * target_actor_vars[i]) for i in range(len(target_actor_vars))]
        update_critic = [target_critic_vars[i].assign(tau * critic_vars[i] + (1 - tau) * target_critic_vars[i]) for i in range(len(target_critic_vars))]

        self.update_target_op = update_actor + update_critic

    def update_target_network(self, sess):
        sess.run(self.update_target_op)
