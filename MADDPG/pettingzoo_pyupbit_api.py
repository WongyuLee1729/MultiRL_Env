import pyupbit
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gym import spaces


# PyUpbit을 사용하여 데이터 가져오기
def get_candle_data(ticker='KRW-BTC', interval='minute1', count=100):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    return df

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

    def dim_info(self):
        dim_info = {}
        for agent_id in self.agents:
            dim_info[agent_id] = []
            dim_info[agent_id].append(self.observation_space[agent_id].shape[0])
            dim_info[agent_id].append(self.action_space[agent_id].n)
        return dim_info

    def set_observation_spaces(self, col_num = 5): # col_num 넘겨 주는 방식 변경할 것 
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
        

    def reset(self,count):
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
        self.current_step = 0
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.agent_selector = agent_selector(self.agents)
        self.agent_selector.reset()
        self.agent_selection = self.agent_selector.next()
        
        self.observations = {agent: 0 for agent in self.agents}
        
        self.set_observation_spaces()
        self.set_action_spaces()
        self.all_done = False
        # dim_val = self.dim_info() 

    def step(self, actions):
        '''
        step(actions) takes in an action for the current agent (specified by agent_selection)
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
            if not self.dones[agent]:
                # 행동에 따라 보상과 관찰값을 다르게 설정
                if action == 0:  # 예: 매수
                    self.rewards[agent] = np.random.randn() * 2  # 예시 보상
                elif action == 1:  # 예: 매도
                    self.rewards[agent] = np.random.randn() * -2  # 예시 보상
                else:  # 행동이 2인 경우: 유지
                    self.rewards[agent] = np.random.randn()  # 예시 보상
                self._cumulative_rewards[agent] += self.rewards[agent]
                

        self.current_step += 1

        if self.current_step >= self.data_len - 1:
            for agent in self.agents:
                self.dones[agent] = True
            self.all_done = True
            
            return self.observations, self.rewards, self.dones, self.infos

        else:
            data = self.candle_data.iloc[self.current_step,:].values
            
            obs = np.array(data)
            
            self.observations = {agent: obs for agent in self.agents} 

            return self.observations, self.rewards, self.dones, self.infos


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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default = 40, help="size of the Data")      
    opt = parser.parse_args()
    
    # 캔들 데이터 가져오기
    candle_data = get_candle_data(count = opt.count) # old version
    
    # 환경 생성
    env = CryptoTradingEnv(candle_data, opt.count)

    # 환경 실행 예제
    env.reset(count = opt.count)
    while not env.all_done:
        # observation = env.observe()
        for agent in env.agent_iter():
            if env.dones[agent]:
                env.step(None)
                continue

            action = {agent_id: env.action_space[agent_id].sample() for agent_id in env.agents}# 에이전트별 행동(0~4)를 선택
            
            next_obs, reward, done, info = env.step(action)
            if env.all_done:
                break
            print(f"Agent: {agent}, Next Observation: {next_obs}, Reward: {reward}, Done: {done}, Info: {info}")
            env.render()
