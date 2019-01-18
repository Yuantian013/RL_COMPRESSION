# RL_COMPRESSION
This project aims to compress some major deep reinforcement learning network.
# Environment
For the test environment, we use OpenAI's cartpole, but make its actions continuous, instead of discrete.
# Dependencies
- Tensorflow (1.9.0)
- OpenAi gym (0.10.8)
# Table of Contents
* Task
  * DDPG(with our tuned hyper parameters, it could converge)
  (cartpole_g10_M1_m0.1_l0.5_tau_0.02_final.ckpt)
  * Compressed DDPG(with our tuned hyper parameters, it could converge)
  (cartpole_g10_M1_m0.1_l0.5_tau_0.02_compression.ckpt)
  * First layer's compression ratio 59.38%
  * Second layer's compression ratio 55.47%
  * PPO(FINISHED)
  (PPO_1.ckpt)
  * Compressed PPO(FINISHED)
  (PPO_compressed_1.ckpt)
  * l1 layer's compression ratio     16.2%
  * mu layer's compression ratio     11.8%
  * sigma layer's compression ratio  26.8%
  * DPPO(FINISHIED)
  (DPPO.ckpt)
  * Compressed DPPO(FINISHIED)
  (DPPO_compressed_1.ckpt)
  * l1 layer's compression ratio     14.5%
  * mu layer's compression ratio     9%
  * sigma layer's compression ratio  51.5%
  * DQN(FINISHED)
  (DQN.ckpt)
  * Compressed DQN(FINISHED)
  (DQN_compressed_1.ckpt)
  * First layer's compression ratio 6.67%
  * Second layer's compression ratio 15.4%
  * Duel_DQN(FINISHED)
  (DQN_dueling.ckpt)
  * Compressed Duel_DQN(FINISHED)
  (DQN_dueling_compressed.ckpt)
  * First layer's compression ratio 15%
  * Value layer's compression ratio 15%
  * Advantage layer's compression ratio 19.2%
  * A3C(FINISHED)
   (A3C.ckpt)
  * Compressed A3C(FINISHED)
  (A3C_compress.ckpt)
  * la layer's compression ratio 19.83%
  * mu layer's compression ratio 12%
  * sigma layer's compression ratio 39%
 
 ## Reference

[1] [Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)  

