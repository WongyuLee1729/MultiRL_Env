import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2

from MADDPG import MADDPG
'''
https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch/blob/master/main.py


torch==1.11.0
PettingZoo[mpe]==1.18.1
numpy==1.22.3
matplotlib==3.5.2
Pillow==9.1.1
gym==0.23.1
pyglet==1.5.15
'''

def get_env(env_name, ep_len=25): # 에피소드의 길이ep_len를 kwargs 형태로 env에 넘겨 줌 
    """
    1. 환경을 정의하고 
    2. 에이전트 별 observation space와 action space를 정의해 줌 
    """
    new_env = None
    if env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(max_cycles=ep_len)

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env')
                        #,choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2'])
    # parser.add_argument('--simple_adversary_v2', type=str,  help='name of the env')    
    
    parser.add_argument('--env_name', default ='simple_adversary_v2', type=str,  help='name of the env')    
    parser.add_argument('--episode_num', type=int, default=30000,help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=100, help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=5e4, help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    args = parser.parse_args()
    
    # create folder to save result
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env, dim_info = get_env(args.env_name, args.episode_length)
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,result_dir)
    
    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}# 에이전트 별로 episode 수 (30000) 사이즈의 array 생성 
    for episode in range(args.episode_num):
        obs = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # 에이전트별 보상을 0으로 초기화(agent reward of the current episode)
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < args.random_steps: 
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}# 에이전트별 행동(0~4)를 선택
            else: # random_steps 이상 step을 진행헀다면..
                action = maddpg.select_action(obs) # 여기서부터 재미있어짐 .. 
                
            next_obs, reward, done, info = env.step(action) # 행동을 환경에 던지고 결과들을 받아옴 (parallel환경이므로 모든 에이전트들이 각각 한번에 받음) 
            # env.render()
            maddpg.add(obs, action, reward, next_obs, done) # buffer memory에 저장
            
            for agent_id, r in reward.items():  # update reward (각 step에 대한 reward를 cumsum해 줌)
                agent_reward[agent_id] += r
                
            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)
                
            obs = next_obs
            
        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r
            
        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            print(message)
            
    maddpg.save(episode_rewards)  # save model
    

    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward


    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {args.env_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))
