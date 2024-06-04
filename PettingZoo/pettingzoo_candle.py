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

    def __init__(self, candle_data):
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
        
        # define observation & action space -> could be defined outside 
        self.observation_spaces = {agent: spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(len(candle_data.columns),), dtype=np.float32) for agent in self.agents}
        self.action_spaces = {agent: spaces.Discrete(3) for agent in self.agents}


    def dim_info(self):
        dim_info = {}
        for agent_id in self.agents:
            dim_info[agent_id] = []
            dim_info[agent_id].append(self.observation_spaces[agent_id].shape[0])
            dim_info[agent_id].append(self.action_spaces[agent_id].n)
        return dim_info


    def observe(self, agent):
        '''
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        '''
        if self.current_step < len(self.candle_data):
            return self.candle_data.iloc[self.current_step].values
        else:
            return np.zeros(self.candle_data.shape[1])  # 빈 관찰값 반환

    def reset(self):
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
        self.all_done = False

    def step(self, action):
        '''
        step(action) takes in an action for the current agent (specified by agent_selection)
        and need to update 
        -reward
        -_cumulative_rewards 
        -terminations
        -truncations
        -infos
        -agent_selection(to the next agent)
        And any internal state used by observe() or render() 
        '''
        if self.all_done:
            return None, None, True, None

        agent = self.agent_selection
        if not self.dones[agent]:
            if self.current_step < len(self.candle_data) - 1: # OHLCV 100개 가져 옴 
                self.current_step += 1
                self.rewards[agent] = np.random.randn()  # 임의의 보상 예시
                self.dones[agent] = False
                self.infos[agent] = {}
            else:
                self.dones[agent] = True

            self._cumulative_rewards[agent] += self.rewards[agent]

        if all(self.dones.values()):
            self.all_done = True
        
        next_obs = self.observe(agent)
        reward = self.rewards[agent]
        done = self.dones[agent]
        info = self.infos[agent]
        self.agent_selection = self.agent_selector.next() # update
        return next_obs, reward, done, info

    def render(self, mode='human'):
        '''
        Renders the environment. In human mode, it can print to terminal, open up a
        graphical window, or open up some other display that a human can see and understand
        '''
        print(f"Step: {self.current_step}")
        for agent in self.agents:
            print(f"{agent} observation: {self.observe(agent)}")

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connection
        or any other environment data which should not be kept around after the users 
        is no longer using the environment
        '''
        pass

    def seed(self, seed=None):
        np.random.seed(seed)

# 캔들 데이터 가져오기
candle_data = get_candle_data()

# 환경 생성
env = CryptoTradingEnv(candle_data)
# dim_info = env.dim_info



# 환경 실행 예제
env.reset()
while not env.all_done:
    for agent in env.agent_iter():
        observation = env.observe(agent)
        if env.dones[agent]:
            env.step(None)
            continue
        action = env.action_spaces[agent].sample()  # 무작위 행동 선택
        next_obs, reward, done, info = env.step(action)
        print(f"Agent: {agent}, Next Observation: {next_obs}, Reward: {reward}, Done: {done}, Info: {info}")
        env.render()
        if env.all_done:
            break  # 모든 에이전트가 done 상태일 때 루프 종료

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











