import numpy as np
import tensorflow as tf
import scipy.io as scio
## DDPG
# reader = tf.train.NewCheckpointReader('Model/cartpole_g10_M1_m0.1_l0.5_tau_0.02_compression.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# Actor_l1_kernel=reader.get_tensor('Actor/l1/kernel')
# Actor_a_kernel=reader.get_tensor('Actor/a/kernel')
#
# scio.savemat('ANALYSE/DDPG_compression',
#              {'Actor_l1_kernel_com': Actor_l1_kernel,
#               'Actor_a_kernel_com': Actor_a_kernel,})
#
# reader = tf.train.NewCheckpointReader('Model/cartpole_g10_M1_m0.1_l0.5_tau_0.02_final.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# Actor_l1_kernel=reader.get_tensor('Actor/l1/kernel')
# Actor_a_kernel=reader.get_tensor('Actor/a/kernel')
#
# scio.savemat('ANALYSE/DDPG',
#              {'Actor_l1_kernel': Actor_l1_kernel,
#               'Actor_a_kernel': Actor_a_kernel,})
## PPO
# reader = tf.train.NewCheckpointReader('Model/PPO_compressed_1.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# a1=reader.get_tensor('pi/dense/kernel')
# a2=reader.get_tensor('pi/dense_1/kernel')
# a3=reader.get_tensor('pi/dense_2/kernel')
# print(a1,a2,a3)
# scio.savemat('ANALYSE/PPO_compression',
#              {'dense_kernel_com': a1,
#               'dense_1_kernel_com': a2,
#               'dense_2_kernel_com': a3})
#
# reader = tf.train.NewCheckpointReader('Model/PPO.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# a1=reader.get_tensor('pi/dense/kernel')
# a2=reader.get_tensor('pi/dense_1/kernel')
# a3=reader.get_tensor('pi/dense_2/kernel')
#
# scio.savemat('ANALYSE/PPO',
#              {'dense_kernel': a1,
#               'dense_1_kernel': a2,
#               'dense_2_kernel': a3})

## DPPO
# reader = tf.train.NewCheckpointReader('Model/DPPO_compress_1.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# a1=reader.get_tensor('pi/dense/kernel')
# a2=reader.get_tensor('pi/dense_1/kernel')
# a3=reader.get_tensor('pi/dense_2/kernel')
# print(a1,a2,a3)
# scio.savemat('ANALYSE/DPPO_compression',
#              {'dense_kernel_com': a1,
#               'dense_1_kernel_com': a2,
#               'dense_2_kernel_com': a3})
#
# reader = tf.train.NewCheckpointReader('Model/DPPO.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# a1=reader.get_tensor('pi/dense/kernel')
# a2=reader.get_tensor('pi/dense_1/kernel')
# a3=reader.get_tensor('pi/dense_2/kernel')
#
# scio.savemat('ANALYSE/DPPO',
#              {'dense_kernel': a1,
#               'dense_1_kernel': a2,
#               'dense_2_kernel': a3})
## DQN
# reader = tf.train.NewCheckpointReader('Model/DQN.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# a1=reader.get_tensor('l1/w1')
# a2=reader.get_tensor('Q/w2')
# scio.savemat('ANALYSE/DQN',
#              {'Q_kernel': a2,
#               'l1_kernel': a1})
# reader = tf.train.NewCheckpointReader('Model/DQN_compressed_1.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# a1=reader.get_tensor('l1/w1')
# a2=reader.get_tensor('Q/w2')
# scio.savemat('ANALYSE/DQN_compression',
#              {'Q_kernel_com': a2,
#               'l1_kernel_com': a1})
## DDQN
# reader = tf.train.NewCheckpointReader('Model/DQN_dueling.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# a1=reader.get_tensor('l1/w1')
# a2=reader.get_tensor('Value/w2')
# a3=reader.get_tensor('Advantage/w2')
# scio.savemat('ANALYSE/DDQN',
#              {'l1': a1,
#               'Value': a2,
#               'Advantage':a3})
# reader = tf.train.NewCheckpointReader('Model/DQN_dueling_compressed.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# a1=reader.get_tensor('l1/w1')
# a2=reader.get_tensor('Value/w2')
# a3=reader.get_tensor('Advantage/w2')
# scio.savemat('ANALYSE/DDQN_compression',
#              {'l1_com': a1,
#               'Value_com': a2,
#               'Advantage_com':a3})
## A3C
reader = tf.train.NewCheckpointReader('Model/A3C_compress.ckpt')
all_variables = reader.get_variable_to_shape_map()

a1=reader.get_tensor('Global_Net/actor/sigma/kernel')
a2=reader.get_tensor('Global_Net/actor/mu/kernel')
a3=reader.get_tensor('Global_Net/actor/la/kernel')
scio.savemat('ANALYSE/A3C_compression',
             {'sigma_com': a1,
              'mu_com': a2,
              'la_com':a3})

reader = tf.train.NewCheckpointReader('Model/A3C.ckpt')
all_variables = reader.get_variable_to_shape_map()

a1=reader.get_tensor('Global_Net/actor/sigma/kernel')
a2=reader.get_tensor('Global_Net/actor/mu/kernel')
a3=reader.get_tensor('Global_Net/actor/la/kernel')
scio.savemat('ANALYSE/A3C',
             {'sigma': a1,
              'mu': a2,
              'la':a3})