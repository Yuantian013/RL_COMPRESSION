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
            learning_rate=0.00001,
            reward_decay=0.9,
            e_greedy=0.9,
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
        self.epsilon = 0.5 if e_greedy_increment is not None else self.epsilon_max

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

        labda = 0.01
        with tf.variable_scope('l1', reuse=True):
            weight_a = tf.get_variable('w1')
        if self.dueling == False:
            with tf.variable_scope('Q', reuse=True):
                weight_b = tf.get_variable('w2')
        else:
            with tf.variable_scope('Value', reuse=True):
                weight_b = tf.get_variable('w2')
            with tf.variable_scope('Advantage', reuse=True):
                weight_c = tf.get_variable('w2')

        if self.dueling==False:
           self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))+labda*(tf.norm(weight_a)+tf.norm(weight_b))
        else:
           self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))+labda*(
                tf.norm(weight_a) + tf.norm(weight_b)+tf.norm(weight_c))


        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

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

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next = self.sess.run(self.q_next, feed_dict={self.S_: batch_memory[:, -self.n_features:]}) # next observation

        q_eval = self.sess.run(self.q_eval, {self.S: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        self.sess.run(self._train_op,feed_dict={self.S: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

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


    def save_result(self):
        if self.dueling==False:
            save_path = self.saver.save(self.sess, "Model/DQN_compressed_1.ckpt")
            print("Save to path: ", save_path)
        else:
            save_path = self.saver.save(self.sess, "Model/DQN_dueling_compressed.ckpt")
            print("Save to path: ", save_path)

    def reload(self):
        if self.dueling==False:
            self.saver.restore(self.sess, "Model/DQN.ckpt")  # 1 0.1 0.5 0.001
            print("Load normal DQN success ")
        else:
            self.saver.restore(self.sess, "Model/DQN_dueling.ckpt")  # 1 0.1 0.5 0.001
            print("Load dueling DQN success ")

MAX_EPISODES=500
MAX_EP_STEPS=200
MEMORY_SIZE = 3000
EWMA_p=0.95
EWMA_step=np.zeros((1,MAX_EPISODES+1))
EWMA_reward=np.zeros((1,MAX_EPISODES+1))
iteration=np.zeros((1,MAX_EPISODES+1))
RENDER = True
DQN_net = DQN(n_actions=25, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=None, dueling=False)
# DQN_net= DQN(n_actions=25, n_features=3, memory_size=MEMORY_SIZE,
#         e_greedy_increment=None, dueling=True)
DQN_net.reload()
for i in range(MAX_EPISODES):
    iteration[0,i+1]=i+1
    s = env.reset()
    ep_reward = 0
    # MAX_EP_STEPS = min(max(500,MAX_EPISODES),1000)
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = DQN_net.choose_action(s)
        f_action = (a - (25 - 1) / 2) / ((25 - 1) / 4)  # [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        s_, r, _, _ = env.step(np.array([f_action]))
        DQN_net.store_transition(s, a, r/10, s_)

        if DQN_net.memory_counter > MEMORY_SIZE:
            DQN_net.learn()

        s = s_
        ep_reward += r
    EWMA_step[0, i + 1] = EWMA_p * EWMA_step[0, i] + (1 - EWMA_p) * j
    EWMA_reward[0, i + 1] = EWMA_p * EWMA_reward[0, i] + (1 - EWMA_p) * ep_reward
    print('Episode:', i, ' Reward: %i' % int(ep_reward), "EWMA_step = ", EWMA_step[0, i + 1], "EWMA_reward = ",
              EWMA_reward[0, i + 1])
    DQN_net.save_result()
DQN_net.save_result()
