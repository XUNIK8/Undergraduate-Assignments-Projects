
# 18数据科学 许函嘉 201800820045 （独立完成）
# 最终参数按作业要求，训练集2000，测试集250，结果见图 "结果"。 另外尝试更改了参数运行了几次，结果见图 "尝试"。

import numpy as np

# 数据预处理函数
def preprocessData(infile,outfile):
    # 说明：
    # 1."教育等级"和"教育时间"效果重复，因此选择删除后者。因此最终一组数据含有 13个feature（5个连续、8个离散）和一个label
    # 2.为了简化代码，先在Excel上对5个连续feature进行了处理，将其分为5类分别命名为 Level0、……、Level4。
    #   并将label命名为 YES（> 50k）和 NO（< 50k）。没有直接赋值为0或1是为了防止在替换字符时发生混乱。
    # 3.将所有feature二值化，最终变为 122 个feature（其中13个为1，其余为0）和 1 个label。以下代码为特征二值化过程。

    # 数组 a 存放所有被替换字符
    a = [
        # 替换5个连续feature（5*5=25个）和 label（1个）
        'Level0','Level1','Level2','Level3','Level4','YES','NO',
        # 替换 workclass（7个）
       'Federal-gov','Local-gov','Private','Self-emp-inc','Self-emp-not-inc','State-gov','Without-pay',
        # 替换 education（16个）
       'One-four','Five-six','Seven-eight','Nine','Ten','Eleven','Twelve','Assoc-acdm','Assoc-voc','HS-grad','Preschool',
       'Prof-school','Some-college','Bachelors','Masters','Doctorate',
        # 替换 marital-status（7个）
       'Never-married','Married-AF-spouse','Married-civ-spouse','Married-spouse-absent','Separated','Divorced',
       'Widowed',
        # 替换 occupation（14个）
       'Adm-clerical','Armed-Forces','Craft-repair','Exec-managerial','Farming-fishing','Handlers-cleaners',
       'Machine-op-inspct','Other-service','Priv-house-serv','Prof-specialty','Protective-serv','Sales','Tech-support',
       'Transport-moving',
        # 替换 relationship（6个）
       'Unmarried','Husband','Not-in-family','Other-relative','Own-child','Wife',
        # 替换 race（5个）
       'Other','Amer-Indian-Eskimo','Asian-Pac-Islander','Black','White',
        # 替换 sex（1个）
       'Female','Male',
        # 替换 native-country（41个）
       'Cambodia','Canada','China','Columbia','Cuba','Dominican-Republic','Ecuador','El-Salvador','England','France',
       'Germany','Greece','Guatemala','Haiti','Holand-Netherlands','Honduras','Hong','Hungary','India','Iran','Ireland',
       'Italy','Jamaica','Japan','Laos','Mexico','Nicaragua','Outlying-US(Guam-USVI-etc)','Peru','Philippines','Poland',
       'Portugal','Puerto-Rico','Scotland','South','Taiwan','Thailand','Trinadad&Tobago','United-States','Vietnam',
       'Yugoslavia',
       ]

    # 数组 b 存放替代原字符的二值化字符
    b = [
        # 替换5个连续feature（5*5=25个）和 label（1个）
        '1,0,0,0,0','0,1,0,0,0','0,0,1,0,0','0,0,0,1,0','0,0,0,0,1','1','-1',
        # 替换 workclass（7个）
       '1,0,0,0,0,0,0','0,1,0,0,0,0,0','0,0,1,0,0,0,0','0,0,0,1,0,0,0','0,0,0,0,1,0,0','0,0,0,0,0,1,0','0,0,0,0,0,0,1',
        # 替换 education（16个）
       '1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0','0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0','0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0','0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0','0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0','0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0','0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0','0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0','0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0','0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0','0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1',
        # 替换 marital-status（7个）
       '1,0,0,0,0,0,0','0,1,0,0,0,0,0','0,0,1,0,0,0,0','0,0,0,1,0,0,0','0,0,0,0,1,0,0','0,0,0,0,0,1,0','0,0,0,0,0,0,1',
        # 替换 occupation（14个）
       '1,0,0,0,0,0,0,0,0,0,0,0,0,0','0,1,0,0,0,0,0,0,0,0,0,0,0,0','0,0,1,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,1,0,0,0,0,0,0,0,0,0,0','0,0,0,0,1,0,0,0,0,0,0,0,0,0','0,0,0,0,0,1,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,1,0,0,0,0,0,0,0','0,0,0,0,0,0,0,1,0,0,0,0,0,0','0,0,0,0,0,0,0,0,1,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,1,0,0,0,0','0,0,0,0,0,0,0,0,0,0,1,0,0,0','0,0,0,0,0,0,0,0,0,0,0,1,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,1,0','0,0,0,0,0,0,0,0,0,0,0,0,0,1',
        # 替换 relationship（6个）
       '1,0,0,0,0,0','0,1,0,0,0,0','0,0,1,0,0,0','0,0,0,1,0,0','0,0,0,0,1,0','0,0,0,0,0,1',
        # 替换 race（5个）
       '1,0,0,0,0','0,1,0,0,0','0,0,1,0,0','0,0,0,1,0','0,0,0,0,1',
        # 替换 sex（1个）
       '0','1',
        # 替换 native-country（41个）
       '1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0',
       '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1',
       ]

    # 生成处理后的新文件
    fin = open(infile, "r")
    fout = open(outfile, "w")

    # 获得数组元素个数、循环替换
    for s in fin:
        if (s.find('?') > -1):
            continue
        else:
            for i in range(len(a)):
                s = s.replace(a[i], b[i])
            fout.write(s)

    fin.close()
    fout.close()

# 将处理后的数据分为 训练集（trainFile） 和 测试架（testFile） 两个文件
def divideData(outfile,trainFile,testFile):
    fin = open(outfile, "r")
    fout1 = open(trainFile, "w")
    fout2 = open(testFile, "w")

    # 处理后的数据约为 30000 组，从 1-30000中随机选取 1800和250个数字，其对应行分别作为 训练集 和 测试集。此过程效果等同于将原数据集 shuffle
    # 设置 seed 以便复现
    np.random.seed(0)
    trainSet = np.random.choice(30000,2000)
    np.random.seed(1)
    testSet = np.random.choice(30000,250)

    # 从原数据集中抽取上述数字对应的行，分别写入 trainFile（训练集） 和 testFile（测试集）
    lines = fin.readlines()
    for i in trainSet:
        fout1.write(lines[i])
    for i in testSet:
        fout2.write(lines[i])

    fin.close()
    fout1.close()
    fout2.close()

# 载入 trainFile（训练集） 中的数据
def loadData(trainFile):
    # labelMat 为每一行的 label（最后一行）组成的矩阵
    # dataMat 为每一行的feature（1-122列）组成的矩阵
    labelMat = []
    dataMat = []
    train = open(trainFile, 'r')
    line = train.readlines()
    for i in range(2000):
        lineArray = line[i].strip().split(',')
        labelMat.append(int(lineArray[-1]))
        dataMat.append(list(map(int, lineArray[0:-1])))
    labelMat = np.array(labelMat)
    dataMat = np.array(dataMat)

    train.close()
    return dataMat, labelMat

# SMO算法中的随机选择 j 的函数
def selectJrand(i, m):
    while True:
        j = int(np.random.uniform(m))
        if j != i:
            return j

# SMO算法中限制 aj 范围的函数
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

# Simplified SMO Algorithm
def smoSimple(dataMat, labelMat, C, tol, max_passes):
    round = 0                      # 记录循环了多少轮，用于运行时检查
    m, n = np.shape(dataMat)       # 获取训练集矩阵的行列数
    passes = 0
    w = np.zeros(n)
    b = 0
    alphas = np.zeros(m)
    alphaold = np.zeros(m)
    alpharaw = np.zeros(m)
    e = np.zeros(m)

    while (passes < max_passes):
        alphaPairsChanged = 0     # 记录 a 改变的个数
        round += 1
        num = 0                   # 用于运行时检查
        for i in range(m):  # 对于每一行数据
            num = num + 1
            print('第', round, '轮', '第', num, '次')    #用于运行书检查
            e[i] = np.sum((alphas) * labelMat * np.dot(dataMat, dataMat[i])) + b - labelMat[i]
            if (labelMat[i] * e[i] < -tol and alphas[i] < C) or (labelMat[i] * e[i] > tol and alphas[i] > 0):
                j = np.random.randint(m)  # 随机选择 aj
                if j != i:
                    e[j] = np.sum((alphas) * labelMat * np.dot(dataMat, dataMat[j])) + b - labelMat[j]
                    alphaold[i] = alphas[i]
                    alphaold[j] = alphas[j]
                    if labelMat[i] != labelMat[j]:
                        L = max(0, alphaold[j] - alphaold[i])
                        H = min(C, C + alphaold[j] - alphaold[i])
                    if labelMat[i] == labelMat[j]:
                        L = max(0, alphaold[j] + alphaold[i] - C)
                        H = min(C, alphaold[j] + alphaold[i])
                    if L == H:
                        continue

                    eta = np.dot(dataMat[i], dataMat[j]) - np.dot(dataMat[i], dataMat[i]) - np.dot(dataMat[j],dataMat[j])
                    if eta >= 0:
                        continue
                    alpharaw[j] = alphaold[j] - labelMat[j] * (e[i] - e[j]) / eta

                    if alpharaw[j] > H:
                        alphas[j] = H
                    if alpharaw[j] < L:
                        alphas[j] = L
                    if L <= alpharaw[j] <= H:
                        alphas[j] = alpharaw[j]
                    if abs(alphas[j] - alphaold[j]) < tol:
                        continue

                    alphas[i] = alphaold[i] + labelMat[i] * labelMat[j] * (alphaold[j] - alphas[j])
                    b1 = b - e[i] - labelMat[i] * (alphas[i] - alphaold[i]) * np.dot(dataMat[i], dataMat[i]) - labelMat[
                        j] * (
                                 alphas[j] - alphaold[j]) * np.dot(dataMat[i], dataMat[j])
                    b2 = b - e[j] - labelMat[i] * (alphas[i] - alphaold[i]) * np.dot(dataMat[i], dataMat[j]) - labelMat[
                        j] * (
                                 alphas[j] - alphaold[j]) * np.dot(dataMat[j], dataMat[j])
                    if 0 < alphas[i] < C:
                        b = b1
                    elif 0 < alphas[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    alphaPairsChanged += 1
        if alphaPairsChanged == 0:
            passes += 1
        else:
            passes = 0
    # 反解 w
    for i in range(m):
        w += labelMat[i] * alphas[i] * dataMat[i]

    return b,alphas,w

# 测试函数
def test(testFile,w,b,C):
    # 获取 testFile（测试集） 中的数据
    num = 0         # 保存 mis-labeled points 的个数
    cost = 0        # cost 值

    labelMat = []
    dataMat = []
    test = open(testFile, 'r')
    line = test.readlines()
    for i in range(250):
        lineArray = line[i].strip().split(',')
        labelMat.append(int(lineArray[-1]))
        dataMat.append(list(map(int, lineArray[0:-1])))
    labelMat = np.array(labelMat)
    dataMat = np.array(dataMat)

    test.close()

    # 利用 w，b 进行测试
    m, n = np.shape(dataMat)
    for i in range(m):
        if (np.dot(w, dataMat[i]) + b) * labelMat[i] < 0:
            # 计算错误数
            num += 1
            # 计算 cost 值（soft SVM）
            cost = float(0.5 * np.dot(w, w) + C * (1 - ((np.dot(w, dataMat[i]) + b) * labelMat[i])))
    print('num = ', num)

    return num, cost

# 主函数
if __name__ == '__main__':
    # 定义所有参数
    infile = "adult.data.raw.txt"
    outfile = "adult.data.preprocessed.txt"
    trainFile = "adult.data.train.txt"
    testFile = "adult.data.test.txt"
    C = 0.05
    tol = 0.001
    max_passes = 1

    # 数据预处理
    preprocessData(infile, outfile)
    # 数据分割
    divideData(outfile, trainFile, testFile)
    # 载入训练集
    dataMat,labelMat = loadData(trainFile)
    # 训练模型
    b,alphas,w = smoSimple(dataMat,labelMat,C, tol, max_passes)
    # 载入测试集
    num,cost = test(testFile,w,b,C)
    # 打印相关结果
    print('C={0}, max_passes={1}, w*w^T={2:.6f}, cost={3:.6f}, number of mis-labeled points={4}'.format(C,max_passes,np.dot(w,w),cost,num))





