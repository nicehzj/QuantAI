import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

class StrategyBase(ABC):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("QuantAI_Gemini")
        self.initial_cash = config['backtest'].get('initial_cash', 20000)
        
    @abstractmethod
    def generate_signals(self, factor_matrix):
        """
        生成交易信号 (截面选股或择时信号)
        factor_matrix: index=[date, symbol], columns=[factors...]
        """
        pass

    def calculate_positions(self, signals, target_count=10):
        """
        根据信号计算目标持仓比例 (简单等权选股逻辑)
        signals: df with index=[date, symbol], column=['signal_score']
        target_count: 每期持仓股票数量
        """
        self.logger.info(f"计算目标持仓，每期选股数量: {target_count}")
        
        # 截面排名：按信号分值从大到小排序
        signals['rank'] = signals.groupby('date')['signal_score'].rank(ascending=False)
        
        # 选出排名前 target_count 的股票
        positions = signals[signals['rank'] <= target_count].copy()
        
        # 设定等权重
        positions['target_weight'] = 1.0 / target_count
        
        return positions[['date', 'symbol', 'target_weight']]

class MultiFactorStrategy(StrategyBase):
    def __init__(self, config, factor_weights=None):
        """
        多因子线性加权策略
        factor_weights: dict, {factor_name: weight}
        """
        super().__init__(config)
        self.factor_weights = factor_weights if factor_weights else {
            'mom_20': 0.4,
            'vol_20': -0.2,   # 低波动率溢价
            'bias_20': -0.2,  # 均值回归
            'vol_ratio': 0.2  # 量价配合
        }

    def generate_signals(self, factor_matrix):
        """
        线性加权合并多个因子得分
        """
        self.logger.info("基于线性加权生成多因子综合评分...")
        
        score_df = factor_matrix.copy()
        score_df['signal_score'] = 0.0
        
        for factor, weight in self.factor_weights.items():
            if factor in score_df.columns:
                score_df['signal_score'] += score_df[factor] * weight
        
        return score_df[['date', 'symbol', 'signal_score']]

if __name__ == "__main__":
    from src.utils.helpers import load_config, setup_logging
    cfg = load_config()
    setup_logging(cfg)
    
    # 示例用法
    # strat = MultiFactorStrategy(cfg)
    # factor_matrix = pd.DataFrame(...) # 加载因子矩阵
    # signals = strat.generate_signals(factor_matrix)
    # positions = strat.calculate_positions(signals, target_count=5)
    # print(positions.head())
