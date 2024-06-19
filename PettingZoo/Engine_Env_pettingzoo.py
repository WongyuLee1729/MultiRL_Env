import pyupbit
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gym import spaces
from simulation_final_data_provider import SimulationDataProvider


# 샘플 멀티에이전트 환경 정의
class CryptoTradingEnv(AECEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, candle_data, count):
        super().__init__()
        '''
        init method는 환경의 arguments들을 받고 다음과 같은 attributes 들을 정의 해야함 
        - possible_agents
        - render_mode
        
        These attributes should not be changed after initialization
        '''
        self.candle_data = candle_data
        self.agents = [f"agent_{i}" for i in range(2)]  # 에이전트 수 예시
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(2))))
        self.data_len = int(count)

    def _dim_info(self,):
        dim_info = {}
        for agent_id in self.agents:
            dim_info[agent_id] = []
            dim_info[agent_id].append(self.observation_space[agent_id].shape[0])
            dim_info[agent_id].append(self.action_space[agent_id].n)
        return dim_info

    def set_observation_spaces(self, col_num = 4): # col_num 넘겨 주는 방식 변경할 것 
        self.observation_space = {agent: spaces.Box(low=-np.inf, high=np.inf,
                shape=(col_num,), dtype=np.float32) for agent in self.agents}
        
    def set_action_spaces(self, action_num= 3): # action_num 넘겨 주는 방식 변경할 것 
        self.action_space = {agent: spaces.Discrete(action_num) for agent in self.agents}


    def observe(self,):
        '''
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        '''

        return self.observations
        

    def reset(self, end_date,count):
        '''
        Reset needs to initialize the following attributes
        -agents
        -rewards
        -_cumulative_rewards
        -terminations
        -truncations
        -infos
        -agent_selection
        And must set up the environment so that render(), step(), and ovserve()
        can be called without issues
        '''
        self.agents = [f"agent_{i}" for i in range(2)]  # 에이전트 수 예시
        self.candle_data.initialize_simulation(end=end_date, count=count)
        self.current_step = 0
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.done = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.agent_selector = agent_selector(self.agents)
        self.agent_selector.reset()
        self.agent_selection = self.agent_selector.next()
        
        
        data = self.candle_data.get_info() # n-1 call
        obs = np.array([data['opening_price'], 
                data['high_price'], 
                data['closing_price'],
                data['acc_volume']])

        self.observations = {agent: obs for agent in self.agents} 

        # self.observations = {agent: 0 for agent in self.agents}
        
        self.set_observation_spaces()
        self.set_action_spaces()
        self.all_done = False
        return self.observations
        # dim_val = self.dim_info() 

    def step(self, actions):
        '''
        step(actions) takes in an action for the current agent 
        (specified by agent_selection)
        and need to update 
        -reward
        -_cumulative_rewards 
        -terminations
        -truncations
        -infos
        -agent_selection(to the next agent)
        And any internal state used by observe() or render() 
        '''

        for agent, action in actions.items():
            if not self.done[agent]:
                # 행동에 따라 보상과 관찰값을 다르게 설정
                if action == 0:  # 예: 매수
                    self.rewards[agent] = np.random.randn() * 2  # 예시 보상
                elif action == 1:  # 예: 매도
                    self.rewards[agent] = np.random.randn() * -2  # 예시 보상
                else:  # 행동이 2인 경우: 유지
                    self.rewards[agent] = np.random.randn()  # 예시 보상
                self._cumulative_rewards[agent] += self.rewards[agent]
                

        self.current_step += 1 
        # !!!!!!! 한번의 에피소드가 끝나는 조건을 코딩해 줄 것 !!!!!!!
        if (self.current_step >= self.data_len - 1) or self.current_step%130 == 0:
            for agent in self.agents:
                self.done[agent] = True  
            self.all_done = True
            # print("1111111111111111111111")
            self.agents = []
            return self.observations, self.rewards, self.done, self.infos

        else:
            data = self.candle_data.get_info() # dr.get_info()
            
            obs = np.array([data['opening_price'], 
                    data['high_price'], 
                    data['closing_price'],
                    data['acc_volume']])
            
            self.observations = {agent: obs for agent in self.agents} 

            return self.observations, self.rewards, self.done, self.infos


    def render(self, mode='human'):
        '''
        Renders the environment. In human mode, it can print to terminal, open up a
        graphical window, or open up some other display that a human can see and understand
        '''
        print(f"Step: {self.current_step}")
        for agent in self.agents:
            print(f"{agent} observation: {self.observe()[agent]}")

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connection
        or any other environment data which should not be kept around after the users 
        is no longer using the environment
        '''
        pass

    def seed(self, seed=None):
        np.random.seed(seed)


if __name__ == "__main__":
    import os 
    import argparse 
    from simulation_final_data_provider import SimulationDataProvider
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--end_date", type=str, default = "2023-04-30T07:30:00", help="End date of data")
    parser.add_argument("--count", type=int, default = 40, help="size of the Data")
    parser.add_argument("--budget", type=int, default = 100000000, help="Budget of money")
    parser.add_argument("--interval", type=int, default = 1, help="inverval of each simulation")
    parser.add_argument("--time_limit", type=int, default = 15, help="simulation time limit in order to manage operation time")     
    # parser.add_argument("--min_data", type=int, default = 5, help="minimum pre-required data to inititate strategy")      
    opt = parser.parse_args()
    end_str = opt.end_date.replace(" ", "T")
    
    # 캔들 from data engine 
    candle_data = SimulationDataProvider("ETH")
    
    
    # 환경 생성
    env = CryptoTradingEnv(candle_data, opt.count)
    # dim_info = env.dim_info


    # 환경 실행 예제
    env.reset(end_date=end_str, count = opt.count)
    while not env.all_done:
        # observation = env.observe()
        for agent in env.agent_iter():
            if env.done[agent]:
                env.step(None)
                continue
            # action = env.action_spaces[agent].sample()  # 무작위 행동 선택
            action = {agent_id: env.action_space[agent_id].sample() for agent_id in env.agents}# 에이전트별 행동(0~4)를 선택
            
            next_obs, reward, done, info = env.step(action)
            if env.all_done:
                break
            print(f"Agent: {agent}, Next Observation: {next_obs}, Reward: {reward}, Done: {done}, Info: {info}")
            env.render()

    
    '''
    Reward 
    
    포트폴리오가치(PV) = 주식 잔고 X 현재 주가 + 현금 잔고 
    
    
    Observation 
    
    - 주식보유비율 = 보유주식수/(포트폴리오가치/현재주가)
    주식 보유 비율은 현재 상태에서 가장 많이 가질 수 있는 주식 수 대비 현재 보유한 주식의 비율임.
    이 값이 0이면 주식을 하나도 보유하지 않은 것이고 0.5면 최대 가질 수 있는 주식 대비 절반의 주식을 보유하고 있는 것이며,
     1이면 최대로 주식을 보유하고 있는 것임. 주식 수가 너무 적으면 매수의 관점에서 투자에 임하고 주식 수가 너무 많으면 매도의 관점에서 투자에 임하게 됨. 
     즉, 보유 주식 수를 투자 행동 결정에 영향을 주기 위해 정책 신경망의 입력에 포함함
    
    - 손익률 = (포트폴리오 가치/ 초기 자본금) -1
    초기 자본금 대비 현재 포트폴리오 가치의 손익률임. 
    포트폴리오 가치는 현재 보유한 주식 평가 가치와 현금 잔고를 합한 값임. 
    보유 주식 수가 적으면 주가가 오르더라도 포트폴리오 가치 손익률은 그보다 적을 것임
    
    - 주당 매수 단가 대비 주가 등락률 = (주가 / 주당 매수 단가) -1
    주당 매수 단가 대비 주가 등략률은 에이전트가 지금까지 매수한 총 대금에 보유 주식 수를 나눈 값인 주당 매수 단가 대비 현재 주가의 위치를 의미함. 
    주당 매수 단가보다 현재 주가가 높으면 수익이 발생함. 보유 주식 비율이 높을수록 현재 손익률과 주당 매수 단가 대비 등략률은 가까워짐.
    
    Action 

    '''
