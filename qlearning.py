"""
1、超参数定义
2、Q-learning策略下的悬崖最优路径规划
   2.1、悬崖创建
   2.2、重置悬崖环境
   2.3、加入路径规划策略
   2.4、训练
   2.5、关闭悬崖环境
3、画图
"""
import random
import numpy as np
import matplotlib.pyplot as plt


# 滑动平均算法
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# 悬崖环境类
class CliffGrid:
    # 初始化悬崖环境参数
    def __init__(self, w=10, h=4, useObstacles=False):
        self.length = w
        self.height = h
        self.useObstacles = useObstacles

        self.action_num = None
        self.init_actions_map()

    # 初始化移动动作及相关参数
    def init_actions_map(self):
        self.act_map = {}
        self.act_map["UP"] = 0
        self.act_map["DOWN"] = 1
        self.act_map["LEFT"] = 2
        self.act_map["RIGHT"] = 3
        self.inv_act_map = {v: k for k, v in self.act_map.items()}
        self.action_num = len(self.act_map)

    # 复制当前网格并返回
    def getObs(self):
        tmp = self.grid.copy()
        tmp[self.agent_loc[0], self.agent_loc[1]] = 2
        return tmp.copy()

    # 获取当前网格状态并输出打印
    def render(self):
        print(self.getObs())

    # 数据类型转换
    def obs2str(self, obs):
        return np.array2string(obs)

    # 生成障碍物位置
    def genObstacles(self, numObstacles, yHigh):
        obstacles = []
        for i in range(0, numObstacles):
            tup = (i + 1, np.random.randint(1, yHigh))
            obstacles.append(tup)
        return obstacles

    # 重置(或重新生成)悬崖环境
    def reset(self):
        # 创建4*10的悬崖网格
        self.grid = np.zeros((self.height, self.length), dtype=np.int32)
        # 对[3,1:9]的位置赋值为1，作为悬崖标志
        self.grid[self.height - 1, 1: self.length - 1] = 1
        # 起点位置
        self.agent_loc = [self.height - 1, 0]
        # 终点位置
        self.goal_loc = [self.height - 1, self.length - 1]

        # 是否对悬崖环境中加入障碍物
        if self.useObstacles:
            # 生成障碍物
            obstacles = self.genObstacles(self.height - 2, self.length - 2)
            # 为悬崖环境添加障碍物
            for obstacle in obstacles:
                self.grid[obstacle[0], obstacle[1]] = 1

        return self.obs2str(self.getObs())

    # 移动策略
    def step(self, action):
        # UP
        if action == 0:
            self.agent_loc[0] -= 1
            if self.agent_loc[0] < 0:
                self.agent_loc[0] = 0
        # DOWN
        elif action == 1:
            self.agent_loc[0] += 1
            if self.agent_loc[0] > self.height - 1:
                self.agent_loc[0] = self.height - 1
        # LEFT
        elif action == 2:
            self.agent_loc[1] -= 1
            if self.agent_loc[1] < 0:
                self.agent_loc[1] = 0
        # RIGHT
        else:
            self.agent_loc[1] += 1
            if self.agent_loc[1] > self.length - 1:
                self.agent_loc[1] = self.length - 1

        # 每正常完成一次移动操作，对奖惩值进行-1
        reward = -1

        # 如果掉入悬崖，则对奖惩值进行-100，并回到初始位置重新开始
        if self.grid[self.agent_loc[0], self.agent_loc[1]] == 1:
            reward = -100
            self.agent_loc = [self.height - 1, 0]

        # 如果到达终点，则将done标志置为True，否则为False
        if self.agent_loc == self.goal_loc:
            done = True
        else:
            done = False

        return (self.obs2str(self.getObs()), reward, done, {})


# Q-learning策略类
class QLearningStrategy:
    def __init__(self, numActs, env, epsilon=0.1, lr=0.01, numIters=10000, discountFactor=0.9, initValue=0.0):
        """
        定义Q-learning决策策略
        :param numActs: 动作数
        :param env: 悬崖环境
        :param epsilon: 贪婪系数
        :param lr: 更新步长
        :param numIters: 迭代次数
        :param discountFactor: 折扣因子
        :param initValue: 初值
        """
        self.initValue = initValue
        self.discountFactor = discountFactor
        self.learningRate = lr
        self.numActs = numActs
        self.numIters = numIters
        self.envObj = env
        self.epsilon = epsilon

    # 创建空的Q表
    def resetDict(self):
        self.qDict = {}

    # 创建动作空间
    def buildDummyActionSpace(self):
        return np.ones(self.numActs).copy() * self.initValue

    # 将创建的动作空间加入Q表中
    def addToDict(self, state):
        if state not in self.qDict.keys():
            self.qDict[state] = self.buildDummyActionSpace()

    # 选取能够使得当前状态Qvector下Q值最大的动作号
    def argmax(self, Qvector):
        if np.count_nonzero(Qvector) == 0:
            action = random.randrange(self.numActs)
        else:
            action = np.argmax(Qvector)
        return action

    # 根据贪婪算法采取动作
    def select_action(self, Qvector):
        if np.random.rand() <= self.epsilon:
            # epsilon randomly choose action
            return random.randrange(self.numActs)
        else:
            # greedily choose action
            return self.argmax(Qvector)

    # 训练
    def learn(self):
        # 创建/重置Q表
        self.resetDict()

        episode = 0
        totalSteps = 0
        self.episodes = []
        self.cumRewards = []
        self.tdErrorList = []
        while totalSteps < self.numIters:
            # 重置悬崖环境
            prevState = self.envObj.reset()
            # 将重置的悬崖环境网格加入Q表
            self.addToDict(prevState)

            cumReward = 0.0
            done = False
            while not done:
                # 根据当前位置选择移动方向
                action = self.select_action(self.qDict[prevState])
                # 由上一步判断的移动方向进行移动
                nextState, reward, done, _ = self.envObj.step(action)
                # 将移动后变化的悬崖环境网格加入Q表
                self.addToDict(nextState)

                # 根据Qlearning算法更新Q表
                bootstrappedTarget = reward + self.discountFactor * np.max(self.qDict[nextState])
                currentEstimate = self.qDict[prevState][action]
                tdError = bootstrappedTarget - currentEstimate
                self.qDict[prevState][action] += self.learningRate * tdError

                # 记录Q表误差及对应的移动奖惩数值
                self.tdErrorList.append(np.absolute(tdError))
                cumReward += reward
                # 记录当前位置
                prevState = nextState
                # 完成一次迭代移动 +1
                totalSteps += 1

                # 如果到达终点，则记录相关信息
                # 并从初始位置重新开始进继续迭代移动
                if done:
                    episode += 1
                    self.episodes.append(episode)
                    self.cumRewards.append(cumReward)
        print("Total States in Q-dict : ", len(self.qDict))

    # 测试
    def execute(self, renderPolicy=False):
        prevState = self.envObj.reset()
        if renderPolicy:
            print("Start State : ")
            self.envObj.render()

        count = 0
        cumReward = 0.0
        done = False
        while not done:
            action = self.argmax(self.qDict[prevState])
            if renderPolicy:
                print("Action : ", self.envObj.inv_act_map[action])
                print('')

            nextState, reward, done, _ = self.envObj.step(action)
            cumReward += reward
            if renderPolicy:
                self.envObj.render()

            prevState = nextState
            count += 1
            if count > 100:
                break

        return cumReward


def main():
    # 1、超参数定义
    grid_w = 10  # 悬崖环境的长
    grid_h = 4  # 悬崖环境的宽

    lr = 0.1  # 更新步长/学习率
    Iters = 100000  # 迭代次数
    epsilon = 0.1  # 贪婪算法的贪婪系数
    discountFactor = 0.5  # 折扣因子 /0.99

    # 是否对悬崖环境添加障碍物，加强模型学习
    Obstacles = True

    # 2、Q-learning策略下的悬崖最优路径规划
    # 加载悬崖
    ClifEnv = CliffGrid(w=grid_w, h=grid_h, useObstacles=Obstacles)
    # 加入路径规划策略
    QLS = QLearningStrategy(numActs=ClifEnv.action_num,
                            env=ClifEnv,
                            epsilon=epsilon,
                            lr=lr,
                            numIters=Iters,
                            discountFactor=discountFactor)

    # 训练Q表
    QLS.learn()

    # 输出训练Q表过程中记录的每轮Q表误差列表
    Errors = QLS.tdErrorList
    # 3、画出每轮Q表的误差曲线
    plt.plot(Errors)
    plt.xlabel("steps")
    plt.ylabel("error in estimates")
    plt.title("Error")
    plt.savefig("Errors.png")
    plt.show()

    # 测试
    QLS.execute(renderPolicy=True)

    Rewards = QLS.cumRewards
    # 3、画出每轮的奖惩值曲线
    plt.plot(Rewards, label="Q-learning")
    plt.xlabel("episodes")
    plt.ylabel("Cum. Reward")
    plt.title("Rewards")
    plt.legend()
    plt.savefig("Rewards.png")
    plt.show()

    # 用滑动平均算法对记录的奖惩值进行平滑处理
    # 使得数据趋势更加明显，易于观察
    RewardsMA = moving_average(QLS.cumRewards, n=200)
    # 3、画出滑动平均后每轮的奖惩值曲线
    plt.plot(RewardsMA, label="Q-learning")
    plt.xlabel("episodes")
    plt.ylabel("Cum. Reward")
    plt.title("Rewards")
    plt.legend()
    plt.savefig("Rewards_MA.png")
    plt.show()


if __name__ == '__main__':
    main()
