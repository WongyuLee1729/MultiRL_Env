from copy import deepcopy
from typing import List
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


def categorical_sample(probs, use_cuda=False):
    int_acs = torch.multinomial(probs, 1)
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    acs = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    return int_acs, acs


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, actor_lr, critic_lr):
        
        self.actor = MLPNetwork(obs_dim, act_dim) # (8,5) ,(10,5), (10,5)
        # critic input all the observations and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        
        # self.critic = MLPNetwork(global_obs_dim, 1) # (43, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        # self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        # self.target_critic = deepcopy(self.critic)

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    # def action(self, obs, model_out=False):
    #     # this method is called in the following two cases:
    #     # a) interact with the environment
    #     # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
    #     # torch.Size([batch_size, state_dim])
    #     # training에서 actor모델 update는 obs + critic에서 전달해주는 값을 받은뒤 취종 action 값을 정함  
    #     logits = self.actor(obs)  # torch.Size([batch_size, action_size])
    #     # action = self.gumbel_softmax(logits)
    #     action = F.gumbel_softmax(logits, hard=True)
    #     if model_out: # critic update 후 actor update을 위해 한번 더 콜 할 때 action & logit 값을 return 해야함
    #         return action, logits
    #     return action


    def action(self, obs, 
                return_all_probs=False, return_log_pi=False,
                regularize=False, return_entropy=False):
        
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])
        # training에서 actor모델 update는 obs + critic에서 전달해주는 값을 받은뒤 취종 action 값을 정함  
        out = self.actor(obs)  # torch.Size([batch_size, action_size])
        probs = F.softmax(out, dim=1)
        int_action, action = categorical_sample(probs)
        rets = [action]
  
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_action))
        if regularize:
            rets.append([(out**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets



    def target_action(self, obs, model_out=False):
        '''when calculate target critic value in MADDPG,
        we use target actor to get next action given next states,
        which is sampled from replay buffer with size torch.Size([batch_size, state_dim])'''

        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        probs = F.softmax(logits, dim=1)
            
        acs_sample = torch.multinomial(probs,1)
        one_hot_act = Variable(torch.FloatTensor(*probs.shape).fill_(0).scatter_(1,acs_sample,1))
        
        if model_out:
            return one_hot_act, acs_sample 
        else:
            return one_hot_act
        
        # action = F.gumbel_softmax(logits, hard=True)
        # return action.squeeze(0).detach()

    # def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
    #     x = torch.cat(state_list + act_list, 1) # -> (1024, 43)
    #     return self.critic(x).squeeze(1)  # tensor with a given length (1024,) -> 출력의 size는 batch_size와 동일하게 해 줌 

    # def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
    #     x = torch.cat(state_list + act_list, 1)
    #     return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    # def update_critic(self, loss):
    #     self.critic_optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
    #     self.critic_optimizer.step()


class MLPNetwork(nn.Module): # Determinstic한 Policy에서 번경해줄것 
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

