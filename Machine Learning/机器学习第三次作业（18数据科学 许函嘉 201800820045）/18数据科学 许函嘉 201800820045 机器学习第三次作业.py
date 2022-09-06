# 18数据科学 许函嘉 201800820045
# 独立完成

import random
random.seed(8)                             #设置随机种子
global Q
Q = {}
for s in range(63):                        #字典初始化Q（s，a）的值  7*9 =63
    for a in range(4):                     # 四个动作
        Q[(s, a)] = 0.0

alpha = 0.9     #步长
gamma = 1       #折扣系数

def epsilon_greedy(state, epsilon):

    action_list = [-9, 9, -1, 1]           # state周围可能执行的动作，分别为上、下、左、右
    action_type = -1                       # 之后用于选取action_list里的动作（0、1、2、3）
    max = -10000                           # 之后用于和Q进行比较，选取最大Q值
    #边角特殊情况处理，0表示不可能执行
    if state == 0:
        action_list[0] = 0
        action_list[2] = 0

    if state == 8:
        action_list[0] = 0
        action_list[3] = 0

    if state == 54:
        action_list[1] = 0
        action_list[2] = 0

    if state == 62:
        action_list[1] = 0
        action_list[3] = 0

    if state == 9 or state == 18 or state == 27 or state == 36 or state == 45:
        action_list[2] = 0

    if state == 17 or state == 26 or state == 35 or state == 44 or state == 53:
        action_list[3] = 0

    if state >= 1 and state <= 7:
        action_list[0] = 0

    if state >= 55 and state <= 61:
        action_list[1] = 0


    #策略选择
    rand_value = random.uniform(0, 1)
    if rand_value < epsilon:                 #生成随机数，大于epsilon，随机选择
        rand_value = random.randint(0, 3)
        action_value = action_list[rand_value]
        action_type = rand_value
    else:                                    #小于epsilon，值函数最大的动作

        for i in range(4):

            if max < Q[(state, i)]:
                max = Q[(state, i)]
                action_value = action_list[i]
                action_type = i

    return action_value, action_type

# SARSA 算法
def SARAS():
    loop_num = 0            # 幕数
    step_num = 0            # 累计步数
    tenTimes_num = 0        # 每一个10局的位数（第 x 个十局）
    tenTimes_valueList = [] # 存放每十局的步数和，共1000个元素

    while loop_num <= 10000:
        #对每幕循环

        step = 0
        loop_num = loop_num + 1
        State = 27                            # 起始位置
        action_s, Action = epsilon_greedy(State, 0.1)     #选策略

        while True:                           #对幕中每一步循环
            step = step + 1                   #执行步数+1
            nextState = State + action_s
            if action_s == 0:                 # 判断agent是否自己撞墙
                Reward = -5
            else:
                Reward = -1

            # 判断风的情况，对state和reward值进行进一步调整
            # 风值为1
            if nextState % 9 == 1 or nextState % 9 == 5 or nextState % 9 == 8:
                if nextState == 1 or nextState == 5 or nextState == 8:
                    Reward = -5
                else:
                    nextState = nextState - 9

            # 风值为2
            if nextState % 9 == 3 or nextState % 9 == 4 or nextState % 9 == 6:
                if nextState == 12 or nextState == 13 or nextState == 15:
                    nextState = nextState - 9
                    Reward = -5
                elif nextState == 3 or nextState == 4 or nextState == 6:
                    Reward = -5
                else:
                    nextState = nextState - 18

            # 进一步判断是否到达终点
            if nextState != 32:
                action_s, nextAction = epsilon_greedy(nextState, 0.1)
                Q[(State, Action)] = Q[(State, Action)] + alpha*(Reward + gamma*Q[(nextState, nextAction)] - Q[(State, Action)])
            else:
                break

            State = nextState
            Action = nextAction

        step_num = step_num + step             # 计算这一幕的总步数

        # 每10局输出一次步数总和
        if loop_num % 10 == 0:
            tenTimes_num = tenTimes_num + 1
            print('第',tenTimes_num,'个十局')   # 打印该10局的位置（第 x 个十局）
            print('步数和:',step_num)           # 打印该10局的步数和
            tenTimes_valueList.append(step_num)         # 将每次的步数和存入一个数组，以便查询最小值
            step_num = 0

    print('10局最小步数和:',min(tenTimes_valueList))     # 打印最小的10局步数和
    print('最小值对应的10局位数:',tenTimes_valueList.index(min(tenTimes_valueList)))  # 打印最小值的位置
    print('\nepisode = ',(tenTimes_valueList.index(min(tenTimes_valueList)) + 1) * 10,'; steps in last 10 episodes = ',min(tenTimes_valueList),'; average steps per episode = ',min(tenTimes_valueList)/10)


if __name__ == '__main__':
    SARAS()
