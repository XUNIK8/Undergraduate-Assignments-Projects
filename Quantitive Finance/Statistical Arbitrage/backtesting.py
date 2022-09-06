import backtrader as bt
import pandas as pd
import time

set_target(['000606.SZ', '000031.SZ'])

# 初始资金
g_init_cash = 10000

class TestStrategy(bt.Strategy):
    def __init__(self):

        # 获取每一只股票的开盘价（open），datas[0]代表000606.SZ，datas[1]代表000031.SZ
        self.dataopen0 = self.datas[0].open
        self.dataopen1 = self.datas[1].open

    def log(self, arg):
        # 打印日志函数，要有这个其他的self.log()才能正常显示
        print('{} {}'.format(self.datetime.date(), arg))

    def next(self):
        # 计算两只股票的差价
        priceGap = self.dataopen0[0] - self.dataopen1[0] * 0.6771655901542087
        # 记录当前价格和差价
        self.log('Close1: %.2f; Close2: %.2f; PriceGap: %.2f' % (self.dataopen0[0], self.dataopen1[0], priceGap))
        # 当前资金
        totalFund = self.broker.getvalue()

        # 判断止盈止亏条件（到达20%收益 / 10%亏损时全部卖出），若达到则结束交易
        if totalFund >= 1.2 * g_init_cash or totalFund <= 0.9 * g_init_cash:
            # 标记1
            self.log('------ END BY PROFIT ------')
            # 记录买入价
            self.log('SELL ALL CREATE, NO1: %.2f; NO2: %.2f' % (self.dataopen0[0], self.dataopen1[0]))

            # 卖出股票1和2
            self.sell(data=self.datas[0], size=self.getposition(self.datas[0]).size)
            self.sell(data=self.datas[1], size=self.getposition(self.datas[1]).size)
            self.stop()

        # 未达到盈亏条件，则进入统计套利策略
        else:
            # 情况一：买入股票1
            if priceGap < 0.5026186255126837:
                # 标记2
                self.log('------ BUY 1 ------')
                # 记录买入价
                self.log('BUY NO.1 CREATE: %.2f' % self.dataopen0[0])

                # 买入股票1
                self.buy(data=self.datas[0], size=100)

            # 情况二：买入股票2
            elif priceGap > 0.7512512512512513:
                # 标记3
                self.log('------ BUY 2 ------')
                # 记录买入价
                self.log('BUY NO.2 CREATE: %.2f' % self.dataopen1[0])

                # 买入股票2
                self.buy(data=self.datas[1], size=100)

            # 情况三：卖出股票1和2（全部），同时结束交易
            elif priceGap > 0.5896400445211823 and priceGap < 0.6642298322427526:
                # 标记4
                self.log('------ END BY STRATEGY ------')
                # 记录买入价
                self.log('SELL ALL CREATE, NO1: %.2f; NO2: %.2f' % (self.dataopen0[0], self.dataopen1[0]))

                # 卖出股票1和2
                self.sell(data=self.datas[0], size=self.getposition(self.datas[0]).size)
                self.sell(data=self.datas[1], size=self.getposition(self.datas[1]).size)
                self.stop()

            # 情况四：不操作
            else:
                # 标记5
                self.log('------ CONTINUE ------')
                return

    def stop(self):
        self.log('------ STOP ------')
        self.log('最终总资产:%.2f元, 盈利:%.2f元' % (
            self.broker.getvalue(), self.broker.getvalue() - g_init_cash))