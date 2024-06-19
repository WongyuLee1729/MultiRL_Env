import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from Agent import Agent
from Buffer import Buffer


def setup_logger(filename):
    """ set up logger with filename. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir):
        # sum all the dims of each agent to get input dim for critic
        global_obs_act_dim = sum(sum(val) for val in dim_info.values()) # critic의 input_dim을 위해 계산함 [8,5], [10,5], [10,5]-> 43 
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, 'cpu')
        self.dim_info = dim_info#{'adversary_0': [8, 5], 'agent_0': [10, 5], 'agent_1': [10, 5]}

        self.batch_size = batch_size
        self.res_dir = res_dir  # directory to save the training result
        self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))

    def add(self, obs, action, reward, next_obs, done):
        '''
        에이전트 별로 obs, action, reward, next_obs, done 등을 buffer memory에 저장해주는 함수
        '''
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.dim_info[agent_id][1])[a] # action을 원핫인코딩해 줌 

            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(o, a, r, next_o, d)

    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers['agent_0'])
        indices = np.random.choice(total_num, size=batch_size, replace=False)# buffer에 쌓아놓은 데이터에서 랜덤하게 batch_size만큼 인덱스를 가져옴 

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            # calculate next_action using target_network and next_state
            next_act[agent_id] = self.agents[agent_id].target_action(n_o)# next_action은 타겟 네트워크에서 가져옴 

        return obs, act, reward, next_obs, done, next_act

    def select_action(self, obs): # obs를 받아 
        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float() # (10,) -> (1,10)
            a = self.agents[agent].action(o)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            actions[agent] = a.squeeze(0).argmax(0) # .item()
            self.logger.info(f'{agent} action: {actions[agent]}')
        return actions

    def learn(self, batch_size, gamma): # critic_value , next_target_critic_value, critic_loss 
        '''
        buffer에서 batch size 만큼의 s,a,r,next_s,next_a를 가져옴 
        
        # Critic update 
        # ==============
        * t시점의 s,a를 먼저 critic neuralnet(1)에 넣어 action-value fcn Q_i를 산출 
        
        * t+1시점의 s,a (next_s,next_a)를 critic neuralnet(2)에 넣어 t+1시점의 Q_j 값을 산출
          -> y = r + gamma * Q_j 로 target 값을 산출 
        
        앞서 구한 두 값을 loss fcn에 넣어 critic을 업데이트 해 줌 
        
        # Actor update 
        # ==============        
        
        '''
    
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size)
            # update critic
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))# obs,act만 넘겨 줌

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(list(next_obs.values()),
                                                                 list(next_act.values())) # next_obs, next_act을 입력 받아 critic 값을 산출 critic_value와 동일한 구조의 뉴럴넷을 사용함 
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs[agent_id], model_out=True)
            act[agent_id] = action
            actor_loss = -agent.critic_value(list(obs.values()), list(act.values())).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            # self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def save(self, reward):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},  # actor parameter
            os.path.join(self.res_dir, 'model.pt')
        )
        with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:  # save training data
            pickle.dump({'rewards': reward}, f)

    @classmethod
    def load(cls, dim_info, file):
        """init maddpg using the model saved in `file`"""
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance
