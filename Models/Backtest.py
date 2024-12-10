import pandas as pd
import logging
import os
from datetime import datetime
class Backtest:
    def __init__(self,OpenPrc,stock_data,StartDay,initial_cash=100000):
        self.Oprc=OpenPrc
        self.startday=StartDay
        self.stock_data = stock_data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = pd.Series(0, index=range(len(stock_data.columns)))
        self.name=stock_data.columns  
        self.history = []  # 记录每个时间步的账户状态


    def set_strategy(self,BS,SS):
        self.BS=BS
        self.SS=SS

    def setlog(self):
        # 每次调用前清除之前的日志处理器
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        if not os.path.exists('log'):
            os.makedirs('log')

        # 获取当前日期并格式化为文件名
        log_filename = f'log/backtest_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

        # 配置日志
        logging.basicConfig(
            filename=log_filename,  # 日志文件名使用日期
            level=logging.INFO,  # 设置日志级别
            format='%(asctime)s - %(levelname)s - %(message)s',  # 日志输出格式
        )

        # 也可以使用 StreamHandler 输出日志到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(console_handler)

        # 在程序中使用 logging
        logging.info('Logging system initialized.')

    def run(self):
        self.day=0
        self.setlog()
        for self.date, row in self.Oprc.loc[self.startday:].iterrows():
            self.prices=row.to_numpy()
            sell=self._sell(self.SS(self.stock_data.loc[:self.date]))
            buy=self._buy(self.BS(self.stock_data.loc[:self.date]))
            self.history.append({
                'date': self.date,
                'cash': self.cash,
                'positions': self.positions.copy(),
                'portfolio_value': self.cash + (self.positions * self.prices).sum(),
                'changed':sell|buy
            })
            logging.info(f"Day {self.day}{'Changed' if self.history[-1]['changed'] else ''} Portfolio_value: {self.history[-1]['portfolio_value']}")
            self.day+=1
    
    def _buy(self, buy_signals):
        for idx, qty in buy_signals.items():
            if qty*self.prices[idx]<self.cash:
                self.cash -= qty*self.prices[idx]
                self.positions.iloc[idx] += qty
        if buy_signals:
            return True
        return False

    def _sell(self, sell_signals):
        for idx, qty in sell_signals.items():
            self.cash -= qty*self.prices[idx]
            self.positions.iloc[idx] += qty
        if sell_signals:
            return True
        return False
  

