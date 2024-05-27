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
        self.candle_data = candle_data
        self.agents = [f"agent_{i}" for i in range(2)]  # 에이전트 수 예시
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(2))))
        self.action_spaces = {agent: spaces.Discrete(3) for agent in self.agents}
        self.observation_spaces = {agent: spaces.Box(low=-np.inf, high=np.inf, shape=(len(candle_data.columns),), dtype=np.float32) for agent in self.agents}
        self.current_step = 0
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.all_done = False

    def observe(self, agent):
        if self.current_step < len(self.candle_data):
            return self.candle_data.iloc[self.current_step].values
        else:
            return np.zeros(self.candle_data.shape[1])  # 빈 관찰값 반환

    def reset(self):
        self.current_step = 0
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selector.reset()
        self.agent_selection = self.agent_selector.next()
        self.all_done = False

    def step(self, action):
        if self.all_done:
            return

        agent = self.agent_selection
        if not self.dones[agent]:
            if self.current_step < len(self.candle_data) - 1:
                self.current_step += 1
                self.rewards[agent] = np.random.randn()  # 임의의 보상 예시
                self.dones[agent] = False
                self.infos[agent] = {}
            else:
                self.dones[agent] = True

            self._cumulative_rewards[agent] += self.rewards[agent]

        if all(self.dones.values()):
            self.all_done = True

        self.agent_selection = self.agent_selector.next()

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        for agent in self.agents:
            print(f"{agent} observation: {self.observe(agent)}")

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)

# 캔들 데이터 가져오기
candle_data = get_candle_data()

# 환경 생성
env = CryptoTradingEnv(candle_data)

# 환경 실행 예제
env.reset()
while not env.all_done:
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        if done:
            continue
        action = env.action_spaces[agent].sample()  # 무작위 행동 선택
        env.step(action)
        env.render()
        if env.all_done:
            break  # 모든 에이전트가 done 상태일 때 루프 종료
