import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn, Tensor
from torch.autograd import Variable
from maac_agent import Agent as maac_agent
from Buffer import Buffer
from copy import deepcopy
# ADDED 
from Attention_critic import AttentionCritic
from New_Attention_critic import AttentionCritic as NewAttentionCritic
MSELoss = torch.nn.MSELoss()

from agents import AttentionAgent



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


def disable_gradients(module):
    for p in module.parameters():
        p.requires_grad = False

def enable_gradients(module):
    for p in module.parameters():
        p.requires_grad = True

class MAAC:
    """MAAC(Actor-Attention-Critic for Multi-Agent Reinforcement Learning) agent"""

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir, 
                 gamma=0.95, tau=0.01, reward_scale=10., pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):
    
        """
        아래 설명 수정 할 것 
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        
        # sum all the dims of each agent to get input dim for critic
        # global_obs_act_dim = sum(sum(val) for val in dim_info.values()) # critic의 input_dim을 위해 계산함 [8,5], [10,5], [10,5]-> 43 
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.maac_agents = {}
        self.buffers = {}

        attention_agents = [] # !!! MAAC old !!!
        
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = maac_agent(obs_dim, act_dim, actor_lr, critic_lr) # actor inside
            
            attention_agents.append([obs_dim, act_dim])  #  !!! MAAC old !!!

            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, 'cpu') # agent 별로 만듦
        
        # =============================================================================
        # Critic init
        # =============================================================================
        self.critic = AttentionCritic(attention_agents) #  !!! MAAC old !!! -> critic 만 있음, MADDPG의 경우 actor와 critic을 묶어 agent class에 정의했었음 
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_critic = deepcopy(self.critic)


        # # =============================================================================
        # # Actor init
        # # =============================================================================
        # self.actor =  MLPNetwork(obs_dim, act_dim)
        # self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        # self.target_actor = deepcopy(self.actor)
        

        self.dim_info = dim_info #{'adversary_0': [8, 5], 'agent_0': [10, 5], 'agent_1': [10, 5]}
        self.batch_size = batch_size
        self.res_dir = res_dir  # directory to save the training result
        self.logger = setup_logger(os.path.join(res_dir, 'maac.log'))


        self.nagents = len(dim_info)
        
        # =============================================================================
        # self.agents = [AttentionAgent(lr=actor_lr,
        #                               hidden_dim=pol_hidden_dim)]
                          #             **params)
                          # for params in agent_init_params] # -> policy/target policy에 대한 뉴럴넷을 만듦  
        # =============================================================================


        # self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
        #                               attend_heads=attend_heads)
        
        # self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
        #                                      attend_heads=attend_heads)
        
        # hard_update(self.target_critic, self.critic) -> 뭔진 모르겠지만 필요없을듯 .. 
        
        # self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
        #                              weight_decay=1e-3)
        
        # self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = actor_lr
        self.q_lr = critic_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0


    # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
    def soft_update(target, source, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
    def hard_update(target, source):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


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
            next_act[agent_id] = self.agents[agent_id].target_action(n_o, model_out=False)# next_action은 타겟 네트워크에서 가져옴 

        return obs, act, reward, next_obs, done, next_act

    def select_action(self, obs): # obs를 받아 Agent.actor에 넣고 action을 받아옴
        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float() # (10,) -> (1,10)
            a = self.agents[agent].action(o)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            actions[agent] = a.squeeze(0).argmax().item()
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
        
        
        Notice
        - critic에서는 모든 agent의 정보를 받아 처리 -> obs, act
        - actor에서는 각 에이전트 별로 처리 -> obs[agent_id]
        '''
        # inps = [] # !!! MAAC old !!!
        soft = True # <- come back later 
        target_policies = []
        for agent_id, agent in self.agents.items(): # actor 개수 만큼 돌아야 하므로 ..=> actor agent들의 시점을 맞추려면 이 loop를 제거해야함 
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size) # buffer에서 데이터를 샘플링해서 가져옴
            # =============================================================================
            # update critic -> Action value parameter update
            # =============================================================================
            # obs, act, reward, next_obs, done = sample
            # Q loss
            next_act = []
            next_log_pis = []
            
            # self.agents[agent_id].target_actor
            for agnt_id in next_obs.keys():
                target_policies.append(self.agents[agnt_id].target_action)
            # 위 아래 for loop 합칠 것 
            for pi, ob in zip(target_policies, next_obs.values()): # target_pol에서 Deterministic Pol에서 categorical로 바꿀것 
                curr_next_ac, curr_next_log_pi = pi(ob, model_out=True) #pi(ob, return_log_pi=True)
                next_act.append(curr_next_ac)
                next_log_pis.append(curr_next_log_pi)
                
            trgt_critic_in = list(zip(next_obs.values(), next_act))
            critic_in = list(zip(obs.values(), act.values()))
 
            next_qs = self.target_critic(trgt_critic_in) # target critic에 입력값을 넣어 줌 
            critic_rets = self.critic(critic_in, regularize=True, niter=self.niter) # critic에 입력값을 넣어 줌 
            
            q_loss = 0
            for a_i, nq, log_pi, (pq, regs) in zip(self.agents.keys(), next_qs,
                                                   next_log_pis, critic_rets):
                target_q = (reward[a_i].view(-1, 1) + self.gamma * nq * (1 - done[a_i].view(-1, 1)))
                
                if soft:
                    target_q -= log_pi / self.reward_scale
                q_loss += MSELoss(pq, target_q.detach())
                for reg in regs:
                    q_loss += reg  # regularizing attention
                    
            q_loss.backward()
            
            self.critic.scale_shared_grads()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), 10 * self.nagents)
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
    
            # if logger is not None:
            #     logger.add_scalar('losses/q_loss', q_loss, self.niter)
            #     logger.add_scalar('grad_norms/q', grad_norm, self.niter)
            self.niter += 1


            # # =============================================================================
            # # update actor -> policy parameter update
            # # =============================================================================
            # # action of the current agent is calculated using its actor
            # action, logits = agent.action(obs[agent_id], model_out=True) # logits = action의 분포값
            # act[agent_id] = action # (1024,5)
            # actor_loss = -agent.critic_value(list(obs.values()), list(act.values())).mean() # gradient ascent 이므로 
            # actor_loss_pse = torch.pow(logits, 2).mean() # 제곱하고 평균 .. 마찬가지로 양수로 바꾸려고 ..?
            # agent.update_actor(actor_loss + 1e-3 * actor_loss_pse) # theta + Alpha * del log(pi_theta) * Q_w(s,a)
            # # self.logger.info(f'{agent_id}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')


            # MAAC actor update 
            samp_act = []
            all_probs = []
            all_log_pis = []
            all_pol_regs = []
    
            for agnt_id in self.agents.keys():
                pi = self.agents[agnt_id].action
                ob = obs[agnt_id]
                curr_ac, probs, log_pi, pol_regs, ent = pi(
                    ob, return_all_probs=True, return_log_pi=True,
                    regularize=True, return_entropy=True)
                # logger.add_scalar('agent%i/policy_entropy' % agnt_id, ent,
                #                   self.niter)
                samp_act.append(curr_ac)
                all_probs.append(probs)
                all_log_pis.append(log_pi)
                all_pol_regs.append(pol_regs)
    
            critic_in = list(zip(obs.values(), samp_act))
            critic_rets = self.critic(critic_in, return_all_q=True)
            
            for agnt_id, probs, log_pi, pol_regs, (q, all_q) in zip(self.agents.keys(), all_probs,
                                                                all_log_pis, all_pol_regs, critic_rets):
                curr_agent = self.agents[agnt_id]
                v = (all_q * probs).sum(dim=1, keepdim=True)
                pol_target = q - v
                if soft:
                    pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
                else:
                    pol_loss = (log_pi * (-pol_target).detach()).mean()
                for reg in pol_regs:
                    pol_loss += 1e-3 * reg  # policy regularization
                # don't want critic to accumulate gradients from policy loss
                disable_gradients(self.critic)
                pol_loss.backward()
                enable_gradients(self.critic)
    
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    curr_agent.actor.parameters(), 0.5)
                curr_agent.actor_optimizer.step()
                curr_agent.actor_optimizer.zero_grad()
    
                # if logger is not None:
                #     logger.add_scalar('agent%i/losses/pol_loss' % a_i,
                #                       pol_loss, self.niter)
                #     logger.add_scalar('agent%i/grad_norms/pi' % a_i,
                #                       grad_norm, self.niter)



    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(self.critic, self.target_critic)

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


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
