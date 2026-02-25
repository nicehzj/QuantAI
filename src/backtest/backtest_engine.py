import pandas as pd
import numpy as np
import logging
from src.utils.helpers import load_config, setup_logging

class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("QuantAI_Gemini")
        self.initial_cash = config['backtest'].get('initial_cash', 20000)

    def run_vectorized_backtest(self, positions, price_df, benchmark_df=None):
        """
        全流程向量化回测，包含 Alpha 计算
        """
        self.logger.info("开始向量化回测 (包含 Alpha 评价)...")
        
        # 1. 计算股票日收益率
        price_df = price_df.copy().sort_values(['symbol', 'date'])
        price_df['pct_chg_calc'] = price_df.groupby('symbol')['close'].pct_change()
        
        # 2. 信号平移 (T+2 执行延迟)
        positions = positions.copy().sort_values(['symbol', 'date'])
        positions['actual_weight'] = positions.groupby('symbol')['target_weight'].shift(2)
        
        # 3. 对齐数据
        price_df['date'] = pd.to_datetime(price_df['date'])
        positions['date'] = pd.to_datetime(positions['date'])
        backtest_df = pd.merge(price_df, positions, on=['date', 'symbol'], how='left')
        backtest_df['actual_weight'] = backtest_df['actual_weight'].fillna(0)
        
        # 4. 计算组合日收益率
        backtest_df['stock_ret'] = backtest_df['pct_chg_calc'] * backtest_df['actual_weight']
        portfolio_daily_ret = backtest_df.groupby('date')['stock_ret'].sum()
        portfolio_nav = (1 + portfolio_daily_ret).cumprod()
        
        # 5. 处理基准收益率 (Alpha 计算)
        benchmark_daily_ret = None
        if benchmark_df is not None:
            benchmark_df = benchmark_df.copy().sort_values('date')
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
            benchmark_df['bench_ret'] = benchmark_df['close'].pct_change()
            # 对齐日期
            common_dates = portfolio_daily_ret.index.intersection(benchmark_df['date'])
            portfolio_daily_ret = portfolio_daily_ret.loc[common_dates]
            benchmark_daily_ret = benchmark_df.set_index('date')['bench_ret'].loc[common_dates]
            portfolio_nav = (1 + portfolio_daily_ret).cumprod()
        
        results = self.calculate_performance(portfolio_daily_ret, portfolio_nav, benchmark_daily_ret)
        return results, portfolio_nav

    def calculate_performance(self, daily_ret, nav_series, bench_ret=None):
        if nav_series.empty: return {}
        
        # 基础指标
        total_ret = nav_series.iloc[-1] - 1
        num_years = len(nav_series) / 242
        annual_ret = (1 + total_ret) ** (1 / num_years) - 1
        
        cum_max = nav_series.cummax()
        max_drawdown = ((nav_series - cum_max) / cum_max).min()
        volatility = daily_ret.std() * np.sqrt(242)
        sharpe = (annual_ret - 0.025) / volatility if volatility != 0 else 0
        
        perf = {
            'Cumulative Return': f"{total_ret:.2%}",
            'Annual Return': f"{annual_ret:.2%}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
        }
        
        # Alpha 指标
        if bench_ret is not None:
            bench_total_ret = (1 + bench_ret).prod() - 1
            bench_annual_ret = (1 + bench_total_ret) ** (1 / num_years) - 1
            
            # Alpha: 策略年化收益 - 基准年化收益
            alpha = annual_ret - bench_annual_ret
            # Beta: 协方差 / 基准方差
            covariance = np.cov(daily_ret, bench_ret)[0][1]
            benchmark_variance = np.var(bench_ret)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
            
            # Tracking Error & Information Ratio
            active_ret = daily_ret - bench_ret
            tracking_error = active_ret.std() * np.sqrt(242)
            ir = alpha / tracking_error if tracking_error != 0 else 0
            
            perf.update({
                'Benchmark Return': f"{bench_total_ret:.2%}",
                'Annual Alpha': f"{alpha:.2%}",
                'Beta': f"{beta:.2f}",
                'Information Ratio': f"{ir:.2f}"
            })
            
        self.logger.info(f"回测绩效统计: {perf}")
        return perf
