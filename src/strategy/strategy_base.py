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
        """
        pass

    def calculate_positions(self, signals, price_df, target_count=20, benchmark_df=None, stop_loss_pct=0.08, min_hold_days=5):
        """
        全自适应 + 强制锁仓版：大幅降低换手率
        """
        self.logger.info(f"开启专家级低换手模式... 止损={stop_loss_pct:.1%}, 最低持仓={min_hold_days}D")
        
        # 1. 识别大盘环境
        signals = signals.copy()
        market_regime = pd.Series('sideways', index=signals['date'].unique())
        if benchmark_df is not None:
            bench = benchmark_df.copy().sort_values('date')
            bench['ma20'] = bench['close'].rolling(20).mean()
            bench['ma60'] = bench['close'].rolling(60).mean()
            
            def get_regime(row):
                if row['close'] > row['ma20'] and row['ma20'] > row['ma60']: return 'bull'
                if row['close'] < row['ma20'] and row['ma20'] < row['ma60']: return 'bear'
                return 'sideways'
            
            regimes = bench.apply(get_regime, axis=1)
            market_regime = pd.Series(regimes.values, index=bench['date'].values)

        # 2. 向量化预计算所有得分 (集成价值因子)
        regime_weights = {
            'bull':     {'mom_60': 0.5, 'mom_20': 0.3, 'pb_val': 0.2},                  # 趋势为主
            'sideways': {'pb_val': 0.6, 'vol_20': -0.2, 'mom_5': -0.2},                 # 低估值+低波动
            'bear':     {'pb_val': 0.8, 'vol_20': -0.2}                                 # 极端价值防御
        }
        
        signals['regime'] = signals['date'].map(market_regime).fillna('sideways')
        signals['final_score'] = 0.0
        for regime, weights in regime_weights.items():
            mask = signals['regime'] == regime
            if mask.any():
                score_part = sum(signals.loc[mask, f] * v for f, v in weights.items() if f in signals.columns)
                signals.loc[mask, 'final_score'] = score_part

        # 3. 预合并价格数据
        signals = pd.merge(signals, price_df[['date', 'symbol', 'close']], on=['date', 'symbol'], how='left').sort_values(['symbol', 'date'])
        # 核心修复：强制数值化
        signals['close'] = pd.to_numeric(signals['close'], errors='coerce')
        
        # 4. Numpy 加速状态机
        from tqdm import tqdm
        all_positions = []
        cols = ['date', 'symbol', 'final_score', 'close', 'regime']
        grouped = signals[cols].groupby('symbol')
        
        for symbol, group in tqdm(grouped, desc="低换手自适应扫描"):
            dates = group['date'].values
            scores = group['final_score'].values
            prices = group['close'].values
            regimes = group['regime'].values
            
            holding = False
            entry_price = 0.0
            hold_days = 0
            weights = np.zeros(len(group))
            
            for i in range(len(group)):
                regime = regimes[i]
                score = scores[i]
                curr_price = prices[i]
                
                # 动态阈值
                buy_trigger = 1.2 if regime == 'bull' else (1.6 if regime == 'sideways' else 2.5)
                sell_trigger = -0.8 if regime == 'bull' else -0.2
                
                if not holding:
                    if regime != 'bear' and score >= buy_trigger:
                        holding = True
                        entry_price = curr_price
                        hold_days = 1
                        weights[i] = 1.0 / target_count
                else:
                    is_stop_loss = entry_price > 0 and curr_price < entry_price * (1 - stop_loss_pct)
                    is_score_drop = score < sell_trigger
                    is_bear_panic = regime == 'bear' and score < 0.5
                    
                    can_exit = (hold_days >= min_hold_days) or is_stop_loss
                    
                    if can_exit and (is_stop_loss or is_score_drop or is_bear_panic):
                        holding = False
                        hold_days = 0
                    else:
                        hold_days += 1
                        weights[i] = 1.0 / target_count
            
            if np.any(weights > 0):
                all_positions.append(pd.DataFrame({'date': dates, 'symbol': symbol, 'target_weight': weights}))

        if not all_positions: return pd.DataFrame(columns=['date', 'symbol', 'target_weight'])
        final_pos = pd.concat(all_positions, ignore_index=True)
        final_pos = final_pos[final_pos['target_weight'] > 0]
        
        daily_sum = final_pos.groupby('date')['target_weight'].sum()
        for d in daily_sum[daily_sum > 1.0].index:
            final_pos.loc[final_pos['date'] == d, 'target_weight'] /= daily_sum[d]
        return final_pos

class MultiFactorStrategy(StrategyBase):
    def __init__(self, config, factor_weights=None):
        super().__init__(config)
        self.factor_weights = factor_weights if factor_weights else {}

    def generate_signals(self, factor_matrix):
        return factor_matrix
