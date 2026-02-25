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
        全流程向量化回测，包含 Alpha 及 胜率/盈亏比 计算
        """
        self.logger.info("开始向量化回测 (包含交易明细统计)...")
        
        # 1. 计算股票日收益率
        price_df = price_df.copy().sort_values(['symbol', 'date'])
        price_df['pct_chg_calc'] = price_df.groupby('symbol')['close'].pct_change()
        
        # 健壮性增强：处理可能的除权错误导致的异常收益率
        # 如果单日涨幅超过 20%，则将其限制为 10% (通常涨停板)
        price_df.loc[price_df['pct_chg_calc'] > 0.20, 'pct_chg_calc'] = 0.10
        price_df.loc[price_df['pct_chg_calc'] < -0.20, 'pct_chg_calc'] = -0.10
        
        # 2. 对齐数据：先 merge，再在真实时间轴上 shift
        price_df['date'] = pd.to_datetime(price_df['date'])
        positions['date'] = pd.to_datetime(positions['date'])
        
        # 创建一个对齐框架，确保所有股票的所有日期都在
        backtest_df = pd.merge(price_df, positions[['date', 'symbol', 'target_weight']], on=['date', 'symbol'], how='left')
        backtest_df['target_weight'] = backtest_df['target_weight'].fillna(0)
        
        # 核心修复：在对齐后的全量时间序列上进行 shift(2)
        # 这样确保 T 日产生的信号在 T+2 日生效，且不会因为数据稀疏导致错位
        backtest_df['actual_weight'] = backtest_df.groupby('symbol')['target_weight'].shift(2).fillna(0)
        
        # 4. 计算组合日收益率
        backtest_df['stock_ret'] = backtest_df['pct_chg_calc'] * backtest_df['actual_weight']
        portfolio_daily_ret = backtest_df.groupby('date')['stock_ret'].sum()
        portfolio_nav = (1 + portfolio_daily_ret).cumprod()

        # --- 核心改进：计算逐笔交易胜率与盈亏比 ---
        # 识别每一笔交易的盈亏 (Trade-level Analysis)
        # 我们定义一笔交易为：一只股票从权重变为非零到归零的过程
        backtest_df['holding'] = (backtest_df['actual_weight'] > 0).astype(int)
        backtest_df['trade_id'] = backtest_df.groupby('symbol')['holding'].diff().abs().cumsum()
        
        # 仅分析持仓期间的数据
        trades = backtest_df[backtest_df['holding'] > 0].copy()
        if not trades.empty:
            # 计算每笔交易的累积收益
            trade_returns = trades.groupby(['symbol', 'trade_id'])['pct_chg_calc'].apply(lambda x: (1 + x).prod() - 1)
            
            num_trades = len(trade_returns)
            winning_trades = trade_returns[trade_returns > 0]
            losing_trades = trade_returns[trade_returns < 0]
            
            win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
            
            avg_win = winning_trades.mean() if not winning_trades.empty else 0
            avg_loss = abs(losing_trades.mean()) if not losing_trades.empty else 0
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else (float('inf') if avg_win > 0 else 0)
        else:
            num_trades = 0
            win_rate = 0
            win_loss_ratio = 0

        # 5. 基准对比
        benchmark_daily_ret = None
        if benchmark_df is not None:
            benchmark_df = benchmark_df.copy().sort_values('date')
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
            benchmark_df['bench_ret'] = benchmark_df['close'].pct_change()
            common_dates = portfolio_daily_ret.index.intersection(benchmark_df['date'])
            portfolio_daily_ret = portfolio_daily_ret.loc[common_dates]
            benchmark_daily_ret = benchmark_df.set_index('date')['bench_ret'].loc[common_dates]
            portfolio_nav = (1 + portfolio_daily_ret).cumprod()
        
        results = self.calculate_performance(portfolio_daily_ret, portfolio_nav, benchmark_daily_ret, 
                                            num_trades, win_rate, win_loss_ratio)
        return results, portfolio_nav

    def calculate_performance(self, daily_ret, nav_series, bench_ret=None, 
                              num_trades=0, win_rate=0, win_loss_ratio=0):
        if nav_series.empty: return {}
        
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
            'Total Trades': num_trades,
            'Win Rate': f"{win_rate:.2%}",
            'Win/Loss Ratio': f"{win_loss_ratio:.2f}"
        }
        
        if bench_ret is not None:
            bench_total_ret = (1 + bench_ret).prod() - 1
            bench_annual_ret = (1 + bench_total_ret) ** (1 / num_years) - 1
            alpha = annual_ret - bench_annual_ret
            perf.update({
                'Benchmark Return': f"{bench_total_ret:.2%}",
                'Annual Alpha': f"{alpha:.2%}",
            })
            
        self.logger.info(f"回测绩效统计: {perf}")
        return perf
