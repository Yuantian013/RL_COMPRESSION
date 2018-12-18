import numpy as np
import tensorflow as tf
import gym
from pendulum_ori import PendulumEnv_ori

np.random.seed(1)
tf.set_random_seed(1)



env = PendulumEnv_ori()
env = env.unwrapped
env.seed(1)

###############################  DQN  ####################################

class DQN(object):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.0001,
            reward_decay=0.9,
            e_greedy=1,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            dueling=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 1 if e_greedy_increment is not None else self.epsilon_max

        self.dueling = dueling  # decide to use dueling DQN or not

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.sess = tf.Session()

        self.S = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.S_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')


        self.q_eval = self._build_net(self.S)# 这个网络是用于及时更新参数
        self.q_next = self._build_net(self.S_, reuse=True)
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        e_params = tf.get_collection('eval_net_params')
        t_params = tf.get_collection('target_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.S: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _build_net(self,s, reuse=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('l1',reuse=reuse):
            n_l1 = 20
            w1 = tf.get_variable('w1', [self.n_features, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

        if self.dueling:
            # Dueling DQN
            with tf.variable_scope('Value',reuse=reuse):
                w2 = tf.get_variable('w2', [n_l1, 1], trainable=trainable)
                b2 = tf.get_variable('b2', [1, 1], trainable=trainable)
                self.V = tf.matmul(l1, w2) + b2

            with tf.variable_scope('Advantage', reuse=reuse,):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], trainable=trainable)
                b2 = tf.get_variable('b2', [1, self.n_actions], trainable=trainable)
                self.A = tf.matmul(l1, w2) + b2

            with tf.variable_scope('Q', reuse=reuse):
                out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)
        else:
            with tf.variable_scope('Q', reuse=reuse):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], trainable=trainable)
                b2 = tf.get_variable('b2', [1, self.n_actions], trainable=trainable)
                out = tf.matmul(l1, w2) + b2
        return out

    def reload(self):
        if self.dueling==False:
            self.saver.restore(self.sess, "Model/DQN_compressed.ckpt")  # 1 0.1 0.5 0.001
            print("Load normal DQN success ")
        else:
            self.saver.restore(self.sess, "Model/DQN_dueling_compressed.ckpt")  # 1 0.1 0.5 0.001
            print("Load dueling DQN success ")

MAX_EPISODES=1000
MAX_EP_STEPS=200
MEMORY_SIZE = 3000
EWMA_p=0.95
EWMA_step=np.zeros((1,MAX_EPISODES+1))
EWMA_reward=np.zeros((1,MAX_EPISODES+1))
iteration=np.zeros((1,MAX_EPISODES+1))
RENDER = True
# DQN_net = DQN(n_actions=25, n_features=3, memory_size=MEMORY_SIZE,
#         e_greedy_increment=None, dueling=False)
DQN_net= DQN(n_actions=25, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=None, dueling=True)
DQN_net.reload()
for i in range(MAX_EPISODES):
    iteration[0,i+1]=i+1
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = DQN_net.choose_action(s)
        f_action = (a - (25 - 1) / 2) / ((25 - 1) / 4)  # [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))
        s_, r, _, _ = env.step(np.array([f_action]))
        s = s_
        ep_reward += r
    EWMA_step[0, i + 1] = EWMA_p * EWMA_step[0, i] + (1 - EWMA_p) * j
    EWMA_reward[0, i + 1] = EWMA_p * EWMA_reward[0, i] + (1 - EWMA_p) * ep_reward
    print('Episode:', i, ' Reward: %i' % int(ep_reward), "EWMA_step = ", EWMA_step[0, i + 1], "EWMA_reward = ",
              EWMA_reward[0, i + 1])
    DQN_net.save_result()
DQN_net.save_result()
