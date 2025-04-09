import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Env_normalize import env
import itertools
import pandas as pd
import openpyxl
import scipy.io as io
import random

# 配置GPU设置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU显存按需增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # 设置可见的GPU设备
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

wb = openpyxl.Workbook()
tf.compat.v1.reset_default_graph()
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
ring = wb.active

from agent_ddpg import DDPG
# from FR-TD3 import DDPG

# 创建Session时配置GPU选项
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 允许GPU显存按需增长
sess = tf.compat.v1.Session(config=config)

def get_agents_action(observation_n, sess, noise_rate):
    actions = []
    for i, obs in enumerate(observation_n):
        agent_name = f'agent{i+1}'
        action = agents[agent_name].action(np.array([obs]), sess) + np.random.randn(1) * noise_rate
        actions.append(action)
    return actions

def train_agent(agent, memory, sess, n, policy_delay):
    batch = memory.sample(n)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
    action_batch = action_batch.reshape(n, 3)
    reward_batch = reward_batch.reshape(n, 1)
    done_batch = done_batch.reshape(n, 1)

    next_actions = agent.target_action(next_state_batch, sess)
    noise = np.clip(np.random.normal(0, 0.2, size=np.shape(next_actions)), -0.5, 0.5)
    next_actions = np.clip(next_actions + noise, -1, 1)

    target_q1 = agent.Q(next_state_batch, next_actions, sess, target=True)
    target_q = reward_batch + 0.9995 * target_q1 * (1 - done_batch)

    target_q = target_q.reshape(-1, 1)  # Ensure target_q has shape (n, 1)
    q_values = agent.train_critic(state_batch, action_batch, target_q, sess)

    agent.train_actor(state_batch, sess)
    agent.update_target_network(sess)
    return q_values

num_agents = 4
rounds = 2000
T = 24
batch_size = 480
noise_rate = 0.2
ep_reward = np.zeros((rounds, T))
all_reward = np.zeros(rounds)
train_point = 0
v_c = np.zeros((num_agents, T))


agent_sequence = np.array([1,2,3,4]) - 1
# agent_sequence = np.array([4,1,3,2]) - 1
# agent_sequence = np.array([1,4,3,2]) - 1
# agent_sequence = np.array([3,1,4,2]) - 1
current_agent_index = 0
current_agent = agent_sequence[current_agent_index]
if __name__ == '__main__':
    with tf.compat.v1.Session(config=config) as sess:  # 使用配置好的session

        agents = {}
        agents_target = {}
        policy_delay = 0

        for i in range(1, 5):
            agent_name = f'agent{i}'
            agents[agent_name] = DDPG(agent_name)

        agent_names = list(agents.keys())
        Num_agents = len(agent_names)

        sess.run(tf.compat.v1.global_variables_initializer())

        for agent in agents.values():
            agent.update_target_network(sess)

        for round in range(10):
            print("round:", round)
            total_reward = 0
            current_states = env.reset()
            for t in range(T):
                current_states = env.get_states()
                actions = get_agents_action(current_states, sess, noise_rate)
                next_states, rewards, done = env.step(actions)
                for i, agent_name in itertools.islice(enumerate(agents), 4):
                    agents[agent_name].memory.add(current_states[i], actions[i], rewards[i],
                                                               next_states[i], done)

        noise_rate = 0.1

        current_agent = 0
        for round in range(rounds):
            total_reward = 0
            current_states = env.reset()
            if round % 200 == 0:
                noise_rate *= 0.8


            for t in range(T):
                current_states = env.get_states()
                actions = get_agents_action(current_states, sess, noise_rate)

                # actions = [np.zeros([1,3]),np.zeros([1,3]),np.zeros([1,3]),np.zeros([1,3])]

                next_states, rewards, done = env.step(actions)
                ep_reward[round, t] = sum(rewards[:])

                for i, agent_name in itertools.islice(enumerate(agents), 4):
                    if i != current_agent:
                        continue
                    agents[agent_name].memory.add(current_states[i], actions[i], rewards[i],
                                                               next_states[i], done)
                    if round >= train_point:
                        train_agent(agents[agent_name], agents[agent_name].memory, sess, batch_size, policy_delay)
                policy_delay += 1

            agents[agent_names[current_agent]].update_target_network(sess)
            # for agent in agents.values():
            #     agent.update_target_network(sess)

            if (round+1) % 50 == 0:
                agents[agent_names[(current_agent + 1) % Num_agents]].copy_parameters(agents[agent_names[current_agent]], sess)
                # current_agent = (current_agent + 1) % Num_agents
                current_agent_index = (current_agent_index + 1) % Num_agents
                current_agent = agent_sequence[current_agent_index]


            all_reward[round] = sum(ep_reward[round,:])
            if round % 1 == 0:
                print(f'Round: {round}; current_agent: {current_agent}; Total Reward: {all_reward[round]}')

        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, "./actor_critic/ddpg_model.ckpt")
        print(f"Model saved in path: {save_path}")