import numpy as np
import tensorflow as tf
import scipy.io as scio
# reader = tf.train.NewCheckpointReader('Save/cartpole_g10_M1_m0.1_l0.5_tau_0.02_compression.ckpt')
# reader = tf.train.NewCheckpointReader('Save/cartpole_g10_M1_m0.1_l0.5_tau_0.02_group_test.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# Actor_l1_kernel=reader.get_tensor('Actor/l1/kernel')
# Actor_a_kernel=reader.get_tensor('Actor/a/kernel')
# Actor_l1_kernel_Adam_1=reader.get_tensor('Actor/l1/kernel/Adam_1')
# Actor_a_kernel=reader.get_tensor('Actor/a/kernel')
# Actor_l1_kernel_Adam=reader.get_tensor('Actor/l1/kernel/Adam')
# Actor_a_kernel=reader.get_tensor('Actor/a/kernel')
# for key in all_variables:
#     print("variable name: ", key)
# print(Actor_a_kernel)
#
# scio.savemat('RNN_CARTPOLE/para_com',
#              {'Actor_l1_kernel_com': Actor_l1_kernel,
#               'Actor_a_kernel_com': Actor_a_kernel,})
#
# import scipy.io as scio
# reader = tf.train.NewCheckpointReader('Save/cartpole_g10_M1_m0.1_l0.5_tau_0.02_final.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# Actor_l1_kernel=reader.get_tensor('Actor/l1/kernel')
# Actor_a_kernel=reader.get_tensor('Actor/a/kernel')
# Actor_l1_kernel_Adam_1=reader.get_tensor('Actor/l1/kernel/Adam_1')
# Actor_a_kernel=reader.get_tensor('Actor/a/kernel')
# Actor_l1_kernel_Adam=reader.get_tensor('Actor/l1/kernel/Adam')
# Actor_a_kernel=reader.get_tensor('Actor/a/kernel')
# for key in all_variables:
#     print("variable name: ", key)
# print(Actor_a_kernel)
#
# scio.savemat('RNN_CARTPOLE/para',
#              {'Actor_l1_kernel': Actor_l1_kernel,
#               'Actor_a_kernel': Actor_a_kernel,})

# reader = tf.train.NewCheckpointReader('Model/PPO_compressed.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# a1=reader.get_tensor('pi/dense/kernel')
# a2=reader.get_tensor('pi/dense_1/kernel')
# a3=reader.get_tensor('pi/dense_2/kernel')
# print(a1)
# print(a2)
# print(a3)
#
# reader = tf.train.NewCheckpointReader('Model/PPO.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# a1=reader.get_tensor('pi/dense/kernel')
# a2=reader.get_tensor('pi/dense_1/kernel')
# a3=reader.get_tensor('pi/dense_2/kernel')
# print(a1)
# print(a2)
# print(a3)
#
#
# reader = tf.train.NewCheckpointReader('Model/PPO_compressed.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# a1=reader.get_tensor('pi/dense/kernel')
# a2=reader.get_tensor('pi/dense_1/kernel')
# a3=reader.get_tensor('pi/dense_2/kernel')
# print(a1)
# print(a2)
# print(a3)

# reader = tf.train.NewCheckpointReader('Model/DQN.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# for key in all_variables:
#     print("variable name: ", key)
# a1=reader.get_tensor('l1/w1')
# a2=reader.get_tensor('Q/w2')
# print(a1)
# print(a2)
# reader = tf.train.NewCheckpointReader('Model/DQN_compressed.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# for key in all_variables:
#     print("variable name: ", key)
# a1=reader.get_tensor('l1/w1')
# a2=reader.get_tensor('Q/w2')
# print(a1)
# print(a2)
# reader = tf.train.NewCheckpointReader('Model/DQN_dueling_compressed.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# for key in all_variables:
#     print("variable name: ", key)

# a1=reader.get_tensor('pi/dense/kernel')
# a2=reader.get_tensor('pi/dense_1/kernel')
# a3=reader.get_tensor('pi/dense_2/kernel')
# print(a1)
# print(a2)
# print(a3)
# reader = tf.train.NewCheckpointReader('Model/DPPO.ckpt')
# all_variables = reader.get_variable_to_shape_map()
# for key in all_variables:
#     print("variable name: ", key)
# a1=reader.get_tensor('pi/dense/kernel')
# a2=reader.get_tensor('pi/dense_1/kernel')
# a3=reader.get_tensor('pi/dense_2/kernel')
# # print(a1)
# # print(a2)
# print(a3)
reader = tf.train.NewCheckpointReader('Model/A3C.ckpt')
all_variables = reader.get_variable_to_shape_map()
for key in all_variables:
    print("variable name: ", key)
a1=reader.get_tensor('Global_Net/actor/sigma/kernel')
a2=reader.get_tensor('Global_Net/actor/mu/kernel')
a3=reader.get_tensor('Global_Net/actor/la/kernel')
print(a1)
print(a2)
print(a3)

reader = tf.train.NewCheckpointReader('Model/A3C_compress.ckpt')
all_variables = reader.get_variable_to_shape_map()
for key in all_variables:
    print("variable name: ", key)
a1=reader.get_tensor('Global_Net/actor/sigma/kernel')
a2=reader.get_tensor('Global_Net/actor/mu/kernel')
a3=reader.get_tensor('Global_Net/actor/la/kernel')
print(a1)
print(a2)
print(a3)

