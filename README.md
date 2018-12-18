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
  * PPO(FINISHED)
  (PPO.ckpt)
  * Compressed PPO(FINISHED)
  (PPO_compressed.ckpt)
  * DPPO(FINISHIED)
  (DPPO.ckpt)
  * Compressed DPPO(FINISHIED)
  (DPPO_compressed.ckpt)
  * DQN(FINISHED)
  (DQN.ckpt)
  * Compressed DQN(FINISHED)
  (DQN_compressed.ckpt)
  * Duel_DQN(FINISHED)
  (DQN_dueling.ckpt)
  * Compressed Duel_DQN(FINISHED)
  (DQN_dueling_compressed.ckpt)
  * A3C(FINISHED)
   (A3C.ckpt)
  * Compressed A3C(FINISHED)
  (A3C_compress.ckpt)
 
 ## Reference

[1] [Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)  

