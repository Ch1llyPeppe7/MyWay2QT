import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
import shutil
import time
from .Backtest import *
from .Utils import *
import torch
import numpy as np
import pandas as pd
# 定义备份目录
BACKUP_DIR = 'strategy_backups'

if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

class Strategy:
    def __init__(self,datadict,startday,initial_money=100000):
        start_time=time.time()
        self.backtest=Backtest(datadict,startday
                              ,initial_money)
        self.backup=datadict['backup']
        print(f"Strategy initialized in {time.time()-start_time} seconds")
    def buy_strategy():
        raise NotImplementedError("Subclasses should implement this method.")
    def sell_strategy():
        raise NotImplementedError("Subclasses should implement this method.")
    def run(self):
        self.backtest.run()
 


    def backup_strategy(self,file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} 不存在，无法备份。")
        
        # 获取文件名和扩展名
        filename = os.path.basename(file_path)
        # 添加时间戳后缀
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_filename = f"{filename.rsplit('.', 1)[0]}_{timestamp}.py"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        
        # 复制文件
        shutil.copy(file_path, backup_path)
        print(f"策略代码已备份到: {backup_path}")

    def show(self):
        # Plot portfolio value
        sns.lineplot(pd.DataFrame(self.backtest.history)['portfolio_value'])

        # Calculate maximum drawdown
        portfolio_values = pd.DataFrame(self.backtest.history)['portfolio_value']
        running_max = portfolio_values.cummax()  # cumulative maximum value
        drawdown = (portfolio_values - running_max) / running_max  # drawdown from running max
        max_drawdown = drawdown.min()  # maximum drawdown value

        # Show maximum drawdown on the plot
        plt.title(f"Portfolio Value - Max Drawdown: {max_drawdown:.2%}")
        plt.show()
        if self.backup:
            self.backup_strategy('Models/Strategy.py')
            
class MyStrategy(Strategy):
    def __init__(self, dfdict, startday, sig,backup,initial_money=100000):
        self.backup=backup
        super().__init__(dfdict, startday, initial_money)
        self.backtest.set_strategy(self.buy_strategy, self.sell_strategy)  # Set strategy methods
        self.sig = sig


    def Refresh_Position(self, Clprc):
        # logging.debug(f"Refreshing positions for day {self.backtest.day}")
        Oprc = self.backtest.prices
        idxs, Yield = SelectStks(Clprc.iloc[:-1])
    
        if idxs.shape[0] == 0:
            return None, None
        
        if self.backtest.day == 0:
            position_idxs = idxs
        else:
            hist_position = self.backtest.positions
            histidxs = torch.tensor(hist_position[hist_position > 0].index, dtype=torch.int32)
            position_idxs = torch.concatenate((histidxs, idxs)).unique()
        
        cov, u = Estimate(Yield, position_idxs)
        p_idxs = position_idxs.cpu().numpy()
        w, sig = cp_maximize_u(cov, u, self.sig)

        now_position = self.backtest.positions.to_numpy()
        total_cash = self.backtest.cash + now_position[p_idxs] @ Oprc[p_idxs]
        
        k = total_cash / (Oprc[p_idxs] @ w)
        W = np.floor(w * k)

        changes = W - now_position[p_idxs]

        return changes, p_idxs

    def buy_strategy(self, Clprc):
        buy_signals = {}
        if self.backtest.day % 30 == 0:
            changes, position_idxs = self.Refresh_Position(Clprc)
            if changes is None:
                return buy_signals
            for idx, qty in zip(position_idxs, changes):
                if qty > 0:
                    buy_signals[idx] = qty
        return buy_signals

    def sell_strategy(self, Clprc):
        sell_signals = {}
        if self.backtest.day % 30 == 0:
            changes, position_idxs = self.Refresh_Position(Clprc)
            if changes is None:
                return sell_signals
            for idx, qty in zip(position_idxs, changes):
                if qty < 0:
                    sell_signals[idx] = qty
                
        else:
            now_position = self.backtest.positions
            now_idxs = now_position.index[now_position > 0]
            for idx, prices in zip(now_idxs, self.backtest.prices[now_idxs]):
                if prices > Clprc.iloc[:, idx].mean() + Clprc.iloc[:, idx].std() or \
                   prices < Clprc.iloc[:, idx].mean() - Clprc.iloc[:, idx].std() / 2:
                    sell_signals[idx] = -now_position[idx]
        return sell_signals

    def run(self):
        super().run()  

    def show(self):
        super().show()  
        logging.info(f"Backtest completed. Portfolio final value: {self.backtest.history[-1]['portfolio_value']}")
        