import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


class AttentionCritic(nn.Module): # 한번에 모든 에이전트 정보를 가져와서 처리함 

   # =============================================================================
   # Original Code
   # =============================================================================
    """
     Attention network, used as critic for all agents. Each agent gets its own
     observation and action, and can also attend over the other agents' encoded
     observations and actions.
     """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
         """
         Inputs:
             sa_sizes (list of (int, int)): Size of state and action spaces per
                                           agent
             hidden_dim (int): Number of hidden dimensions
             norm_in (bool): Whether to apply BatchNorm to input
             attend_heads (int): Number of attention heads to use (use a number
                                 that hidden_dim is divisible by)
         """
         super(AttentionCritic, self).__init__()
         assert (hidden_dim % attend_heads) == 0
         self.sa_sizes = sa_sizes
         self.nagents = len(sa_sizes)
         self.attend_heads = attend_heads
         
         self.critic_encoders = nn.ModuleList() # 1. state, action에 대한 encoding
         self.critics = nn.ModuleList()         # 2. Q값 산출(f_i)  -> Q_i^{psi} = f_i(g_i(o_i,a_i),x_i)
         self.state_encoders = nn.ModuleList()  # 3. state에 대한 encoding -> e_i = g_i(obs, act)
         
         # iterate over agents
         for sdim, adim in sa_sizes:
             idim = sdim + adim
             odim = adim
             
             encoder = nn.Sequential()
             if norm_in:
                 encoder.add_module('enc_bn', nn.BatchNorm1d(idim,affine=False))
             encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
             encoder.add_module('enc_nl', nn.LeakyReLU())
             self.critic_encoders.append(encoder)
             
             critic = nn.Sequential()
             critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,hidden_dim))
             critic.add_module('critic_nl', nn.LeakyReLU())
             critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
             self.critics.append(critic)
             
             state_encoder = nn.Sequential()
             if norm_in:
                 state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(sdim, affine=False))
             state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,hidden_dim))
             state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
             self.state_encoders.append(state_encoder)
             
         attend_dim = hidden_dim // attend_heads
         self.key_extractors = nn.ModuleList()      # Key
         self.selector_extractors = nn.ModuleList() # Query
         self.value_extractors = nn.ModuleList()    # Value
         
         for i in range(attend_heads):
             self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
             self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
             self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,attend_dim),nn.LeakyReLU()))

         self.shared_modules = [self.key_extractors, self.selector_extractors,
                                self.value_extractors, self.critic_encoders]


    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
                                                 
                                                 
        all_q : (1024,5) 사이즈의 q를 list로 전부 저장할 것인지 
        regularize : regularization을 할 것인지 
        return_attend : attention weight를 저장할 것인지 
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps] # (1024,8) , (1024,10), (1024,10) ..
        actions = [a for s, a in inps]  # (1024,5) , (1024,5), (1024,5) ..
        inps = [torch.cat((s, a), dim=1) for s, a in inps] # (1024,13) , (1024,15), (1024,15) ..
        
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]  
        # -> output size : [1024 x 32], [1024 x 32], [1024 x 32] 
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents] # g_i(o_i, a_i)= e_i 계산
        # s_encodings = self.state_encoders[agents]
        
        
        # extract keys for each head for each agent -> key는 Atten head 만큼만들었음 
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors] 
        
        # extract sa values for each head for each agent -> value는 Atten head 만큼만들었음 
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        
        # extract selectors for each head for each agent that we're returning Q for -> Atten head 만큼 만듦
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        
        # Multi-head Attention 
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors): # calculate attention per head
            
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i] # e_j 
                values = [v for j, v in enumerate(curr_head_values) if j != a_i] # e_j
                
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2) # softmax
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2) # x_i = alpha*v_j
                other_all_values[i].append(other_values) # x_i
                all_attend_logits[i].append(attend_logits) 
                all_attend_probs[i].append(attend_weights)
                
        # calculate Q per agent-> Q_i^{psi}(o,a) = f_i(g_i(o_i,a_i),x_i)
        all_rets = []
        for i, a_i in enumerate(agents): # 에이전트 별로 돌면서 Q를 계산
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]] # logger에 넣는놈인대 잘 모르겠음 .. 
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1) #concat per agent (1024,32+32)
            all_q = self.critics[a_i](critic_in) # Q_i^{psi} = f_i(g_i(o_i,a_i),x_i) , (1024,5)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1] # action은 max를 가져옴 
            q = all_q.gather(1, int_acs) # .gather -> 특정 인덱스 값만 산출하는 함수 , all_q에서 int_acs에 맞는 값만 가져옴 
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets # 조건에 따라 return_q 또는 return_all_q를 산출함
