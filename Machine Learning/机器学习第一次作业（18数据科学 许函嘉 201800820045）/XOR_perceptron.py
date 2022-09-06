
# 18数据科学 许函嘉 201800820045（个人完成）

# 主函数前为perceptron算法部分，主函数为具体对XOR执行perceptron
# 经训练后（学习率0.25，次数100），最终可以打印成功组数和所有成功组合。

import numpy as np
outcome = []        #用来存放成功组合
num = 0             #用来记录成功组数
class pcn:

    def __init__(self, inputs, targets):
        if np.ndim(inputs) > 1:
            self.nIn = np.shape(inputs)[1]
        else:
            self.nIn = 1

        if np.ndim(targets) > 1:
            self.nOut = np.shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = np.shape(inputs)[0]
        self.weights = np.random.rand(self.nIn + 1, self.nOut) * 0.1 - 0.05

    # 训练更新权重
    def pcntrain(self, inputs, targets, eta, nIterations):

        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        change = range(self.nData)

        for n in range(nIterations):
            self.activations = self.pcnfwd(inputs);
            self.weights -= eta * np.dot(np.transpose(inputs), self.activations - targets)

    # 比较触发函数
    def pcnfwd(self, inputs):

        activations = np.dot(inputs, self.weights)
        return np.where(activations > 0, 1, 0)

    # 比较输出值和目标值、判断是否成功
    def confmat(self, inputs, targets):
        global num
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        outputs = np.dot(inputs, self.weights)
        nClasses = np.shape(targets)[1]

        if nClasses == 1:
            nClasses = 2
            outputs = np.where(outputs > 0, 1, 0)
        else:
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        # 输出每一个组合的相关结果
        print('输出值:')
        print(np.transpose(outputs))
        print('目标值:')
        print(np.transpose(targets))
        print('Gram矩阵:')
        print (cm)
        print ('正确率:%f' %(np.trace(cm) / np.sum(cm)))
        if np.trace(cm) / np.sum(cm) == 1:
            print('结果：SUCCEED\n')
            outcome.append(inputs[:,2])                    #将新成功的[a,b,c,c]组合拼接到”成功组合数组“中
            num += 1                                       #每成功一组，总数计数+1
        else:
            print('结果：FAILED\n')

    def outcome(self):
        print('综上,成功组数为：%d ; 所有成功组合[a,b,c,d]为：\n' %num) #输出成功总数num
        print(outcome)                                            #输出所有成功组合



# 主函数XOR
if __name__ == '__main__':
# 调用pcn文件里的perceptron算法
    from XOR_perceptron import pcn

# 假设添加的一维向量为 [a,b,c,d]，则新的XOR真值表如下：a1-a16
    a1 = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0]])
    a2 = np.array([[0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0]])
    a3 = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0]])
    a4 = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 1], [1, 1, 0, 0]])
    a5 = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0]])
    a6 = np.array([[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0]])
    a7 = np.array([[0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 1], [1, 1, 0, 0]])
    a8 = np.array([[0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0]])
    a9 = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]])
    a10 = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 0]])
    a11 = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 0]])
    a12 = np.array([[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]])
    a13 = np.array([[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 0]])
    a14 = np.array([[0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 0]])
    a15 = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 0]])
    a16 = np.array([[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 0]])

# 依次对16组不同的输入组合进行perceptron，依次显示每一组的 1.输出值 2.目标值 3.Gram矩阵 4.正确率 。最后成功的打印“SUCCEED”，失败的打印“FAILED”
    print('第1组：a,b,c,d = [0,0,0,0]')
    p = pcn(a1[:, 0:3], a1[:, 3:])
    p.pcntrain(a1[:, 0:3], a1[:, 3:], 0.25, 100)
    p.confmat(a1[:, 0:3], a1[:, 3:])

    print('第2组：a,b,c,d = [1,0,0,0]')
    p = pcn(a2[:, 0:3], a2[:, 3:])
    p.pcntrain(a2[:, 0:3], a2[:, 3:], 0.25, 100)
    p.confmat(a2[:, 0:3], a2[:, 3:])

    print('第3组：a,b,c,d = [0,1,0,0]')
    p = pcn(a3[:, 0:3], a3[:, 3:])
    p.pcntrain(a3[:, 0:3], a3[:, 3:], 0.25, 100)
    p.confmat(a3[:, 0:3], a3[:, 3:])

    print('第4组：a,b,c,d = [0,0,1,0]')
    p = pcn(a4[:, 0:3], a4[:, 3:])
    p.pcntrain(a4[:, 0:3], a4[:, 3:], 0.25, 100)
    p.confmat(a4[:, 0:3], a4[:, 3:])

    print('第5组：a,b,c,d = [0,0,0,1]')
    p = pcn(a5[:, 0:3], a5[:, 3:])
    p.pcntrain(a5[:, 0:3], a5[:, 3:], 0.25, 100)
    p.confmat(a5[:, 0:3], a5[:, 3:])

    print('第6组：a,b,c,d = [1,1,0,0]')
    p = pcn(a6[:, 0:3], a6[:, 3:])
    p.pcntrain(a6[:, 0:3], a6[:, 3:], 0.25, 100)
    p.confmat(a6[:, 0:3], a6[:, 3:])

    print('第7组：a,b,c,d = [1,0,1,0]')
    p = pcn(a7[:, 0:3], a7[:, 3:])
    p.pcntrain(a7[:, 0:3], a7[:, 3:], 0.25, 100)
    p.confmat(a7[:, 0:3], a7[:, 3:])

    print('第8组：a,b,c,d = [1,0,0,1]')
    p = pcn(a8[:, 0:3], a8[:, 3:])
    p.pcntrain(a8[:, 0:3], a8[:, 3:], 0.25, 100)
    p.confmat(a8[:, 0:3], a8[:, 3:])

    print('第9组：a,b,c,d = [0,1,1,0]')
    p = pcn(a9[:, 0:3], a9[:, 3:])
    p.pcntrain(a9[:, 0:3], a9[:, 3:], 0.25, 100)
    p.confmat(a9[:, 0:3], a9[:, 3:])

    print('第10组：a,b,c,d = [0,1,0,1]')
    p = pcn(a10[:, 0:3], a10[:, 3:])
    p.pcntrain(a10[:, 0:3], a10[:, 3:], 0.25, 100)
    p.confmat(a10[:, 0:3], a10[:, 3:])

    print('第11组：a,b,c,d = [0,0,1,1]')
    p = pcn(a11[:, 0:3], a11[:, 3:])
    p.pcntrain(a11[:, 0:3], a11[:, 3:], 0.25, 100)
    p.confmat(a11[:, 0:3], a11[:, 3:])

    print('第12组：a,b,c,d = [1,1,1,0]')
    p = pcn(a12[:, 0:3], a12[:, 3:])
    p.pcntrain(a12[:, 0:3], a12[:, 3:], 0.25, 100)
    p.confmat(a12[:, 0:3], a12[:, 3:])

    print('第13组：a,b,c,d = [1,1,0,1]')
    p = pcn(a13[:, 0:3], a13[:, 3:])
    p.pcntrain(a13[:, 0:3], a13[:, 3:], 0.25, 100)
    p.confmat(a13[:, 0:3], a13[:, 3:])

    print('第14组：a,b,c,d = [1,0,1,1]')
    p = pcn(a14[:, 0:3], a14[:, 3:])
    p.pcntrain(a14[:, 0:3], a14[:, 3:], 0.25, 100)
    p.confmat(a14[:, 0:3], a14[:, 3:])

    print('第15组：a,b,c,d = [0,1,1,1]')
    p = pcn(a15[:, 0:3], a15[:, 3:])
    p.pcntrain(a15[:, 0:3], a15[:, 3:], 0.25, 100)
    p.confmat(a15[:, 0:3], a15[:, 3:])

    print('第16组：a,b,c,d = [1,1,1,1]')
    p = pcn(a16[:, 0:3], a16[:, 3:])
    p.pcntrain(a16[:, 0:3], a16[:, 3:], 0.25, 100)
    p.confmat(a16[:, 0:3], a16[:, 3:])

# 综上，输出最终结果（打印成功组数和所有成功组合）
    p.outcome()




