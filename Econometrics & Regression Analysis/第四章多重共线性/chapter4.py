
# 四. 多重线性回归

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 读取数据

data = pd.read_csv('./data.csv', encoding =  'UTF-8')
# data = pd.read_excel('./data_1.xlsx')
print(data.head(5))

# 提取因变量和自变量

# data.csv
X = data.iloc[:, 0:4]
Y = data.iloc[:, 5]
# data_1.xlsx
# X = data.iloc[:, 2:]
# Y = data.iloc[:, 1]

## 2. 判断多重共线性

# ols法估计，R^2值高、F检验值高、且x1,x2，x3的t检验不显著

X1 = sm.add_constant(X) #加上一列常数1，这是回归模型中的常数项
reg = sm.OLS(Y, X1) #生成回归模型
model = reg.fit() #拟合数据
print(model.summary())

# 相关系数，对数据进行标准化处理（z-score标准化），可见有共线性

X = (X - X.mean())/np.std(X)
Y = (Y - Y.mean())/np.std(Y)
print(X.corr())

# 分割数据

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.7, random_state=1)

# 3. 消除多重共线性（PCA法）

# 对模型进行训练，返回降维后数据

pca = PCA(n_components='mle')
pca.fit(X_train)
X_train = pca.transform(X_train)
Y_train= (Y_train - Y_train.mean())/np.std(Y)
print(X_train)

# 4. 重建线性回归

# 使用返回后的数据用线性回归模型建模，ols回归后R^2为0.933，p值小，说明模型拟合效果好

import statsmodels.api as sm
ols = sm.OLS(Y_train, X_train).fit()
print(ols.summary())

pca.explained_variance_ratio_

print(pca.get_params())

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
lr.score(X_train, Y_train)

# 5. 测试集验证

X_test = (X_test - X_test.mean())/np.std(X_test)
X_test = pca.transform(X_test)
print(X_test)

# 预测值
y_pred = lr.predict(X_test)
print(y_pred)

# 真实值
print(Y_test)

# 比较真实值与预测值

plt.scatter(y_pred, Y_test)
plt.xlabel('The predicted Y of LinearRegression')
plt.ylabel('The real Y')
plt.show()

# R^2值为0.868，说明在测试集上回归效果较好，也说明PCA方法较好地消除了多重共线性

olsr = sm.OLS(y_pred, Y_test).fit()
print(olsr.summary())
