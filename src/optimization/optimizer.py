import pandas as pd
import numpy as np
import logging
import itertools
from src.strategy.strategy_base import MultiFactorStrategy
from src.backtest.backtest_engine import BacktestEngine

class StrategyOptimizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("QuantAI_Gemini")
        self.engine = BacktestEngine(self.config)

    def grid_search_weights(self, factor_matrix, price_df, weight_ranges, benchmark_df=None):
        """
        全量网格搜索寻优，并找出最佳因子贡献
        """
        self.logger.info("启动高级网格搜索寻优...")
        factors = list(weight_ranges.keys())
        weight_combinations = list(itertools.product(*weight_ranges.values()))
        
        best_metric = -np.inf
        best_report = None
        
        # 结果汇总表
        summary_results = []

        from tqdm import tqdm
        for weights in tqdm(weight_combinations, desc="寻优进度"):
            current_weights = dict(zip(factors, weights))
            
            # 运行策略 (专家模式：买入 1.2, 卖出 0.0, 止损 8%, 择时开启)
            strat = MultiFactorStrategy(self.config, factor_weights=current_weights)
            signals = strat.generate_signals(factor_matrix)
            positions = strat.calculate_positions(
                signals, 
                price_df, 
                target_count=20, 
                buy_threshold=1.2, 
                sell_threshold=0.0,
                stop_loss_pct=0.08,
                benchmark_df=benchmark_df
            )
            
            # 运行回测
            perf, _ = self.engine.run_vectorized_backtest(positions, price_df, benchmark_df=benchmark_df)
            
            if not perf: continue
            
            # 以夏普比率作为核心寻优指标，结合年化 Alpha 综合评分
            sharpe = float(perf['Sharpe Ratio'])
            
            # 安全解析 Alpha 字符串 (处理 25.5% -> 0.255)
            alpha_str = perf.get('Annual Alpha', '0%').replace('%', '')
            try:
                alpha_val = float(alpha_str) / 100.0
            except:
                alpha_val = 0.0
            
            # 综合得分：夏普比率权重 70% + 超额收益权重 30%
            # 这是量化常用的评分模型，兼顾风险和收益
            score = (sharpe * 0.7) + (alpha_val * 10.0 * 0.3)
            
            result_entry = {
                'weights': current_weights,
                'sharpe': sharpe,
                'alpha': alpha_val,
                'total_trades': perf['Total Trades'],
                'annual_ret': perf['Annual Return'],
                'score': score
            }
            summary_results.append(result_entry)
            
            if score > best_metric:
                best_metric = score
                best_report = result_entry
                
        # 找出最优因子：在最佳权重组合中，绝对值最大的权重对应的因子
        best_factor = "None"
        if best_report:
            w_dict = best_report['weights']
            best_factor = max(w_dict, key=lambda k: abs(w_dict[k]))

        self.logger.info(f"寻优完成。最佳因子: {best_factor}")
        return best_report, best_factor, pd.DataFrame(summary_results)
