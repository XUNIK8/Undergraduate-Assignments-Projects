import pandas as pd
import quantmind as qm
import numpy as np
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="whitegrid")

#数据选取2020年一整年的数据
trading_date=qm.get_trade_date("20200101","20201231")

stock_list=qm.get_stock_list()['code'][:200]

# 此处考虑到服务器性能 与 运行时间。 只选取400只股票作为股票池 
total_data=pd.DataFrame([])
for stock_code in stock_list:
    total_data[stock_code]=qm.get_bar(stock_code, start_date='20200101', end_date='20201231', freq='60m')['close']

#在获取所有数据后，在寻找交易对之前，要验证每一只股票数据的平稳性，采用ADF检验
# 由于实际数据波动性较大，这里P指的阈值放宽一些
def stationarity_test(data, cutoff=0.1):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(data)[1]
    if pvalue < cutoff:
        print('p-value = ' + str(pvalue) + ' The series ' + data.name +' is likely stationary.')
        return True
    else:
        print('p-value = ' + str(pvalue) + ' The series ' + data.name +' is likely non-stationary.')
        return False

stationarity_data=pd.DataFrame([])
for stock_code in stock_list:
    data=total_data[stock_code]
    if stationarity_test(data):
        stationarity_data[stock_code]=data

# 下一步进行 协整性测试 找出有协整性的 

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

scores, pvalues, pairs = find_cointegrated_pairs(stationarity_data)
tickers=stationarity_data.keys()
import seaborn
fig, ax = plt.subplots(figsize=(51,51))
seaborn.heatmap(pvalues, xticklabels=tickers, yticklabels=tickers, cmap='RdYlGn_r' ,mask = (pvalues >= 0.05))
plt.show()

corr_dict={}
for pair in pairs:
    stock1_data=stationarity_data[pair[0]]
    stock2_data=stationarity_data[pair[1]]
    corr_dict[pair]=stock1_data.corr(stock2_data)

# 选出相关性最大的一组 就是我们策略需要的交易对  他同时满足 平稳 协整 相关三个条件
print(max(corr_dict.values()))
#print(corr_dict[max(corr_dict)])

stock1=stationarity_data['000031.SZ']
stock2=stationarity_data['000606.SZ']
plt.plot(stock1)
plt.plot(stock2)
plt.show()

stock1 = sm.add_constant(stock1)
results = sm.OLS(stock2, stock1).fit()
stock1=stock1['000031.SZ']
b = results.params['000031.SZ']

spread = stock2 - b * stock1
spread.plot()
plt.axhline(spread.mean(), color='black')
plt.legend(['Spread'])
plt.show()

spread_mean=spread.mean()

# 用数学的方法确定阈

from scipy.stats import norm
from scipy.signal import find_peaks
def profit(threshold,T=len(trading_date)):
    return T*threshold*(1-norm.cdf(threshold))

from scipy import optimize
maximum = optimize.fminbound(profit, -0.5, 1.5)

# 网格搜索 确定最大值点 买入价差
X=np.linspace(-0.5,1.5,1000)
Y=list(map(profit,X))

buy_threhold1=X[np.argmax(Y)]
buy_threhold2=2*spread_mean-buy_threhold1

sell_threhold=(spread_mean-0.3*(spread_mean-buy_threhold2),spread_mean+0.3*(buy_threhold1-spread_mean))