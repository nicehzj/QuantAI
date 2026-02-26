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
        
        # 1. 预处理数据
        price_df = price_df.copy().sort_values(['symbol', 'date'])
        price_df['date'] = pd.to_datetime(price_df['date'])
        positions['date'] = pd.to_datetime(positions['date'])
        
        # 2. 核心改进：修正信号时空错位 (先对齐全量行情，再在连续时间轴上平移)
        backtest_df = pd.merge(price_df, positions[['date', 'symbol', 'target_weight']], on=['date', 'symbol'], how='left')
        backtest_df['target_weight'] = backtest_df.groupby('symbol')['target_weight'].ffill().fillna(0)
        
        # 3. 计算并清洗股票日收益率 (遵循 A 股涨跌停规则)
        backtest_df['pct_chg_calc'] = backtest_df.groupby('symbol')['close'].pct_change()
        
        # 定义 A 股差异化涨跌停限制
        def get_limit(symbol):
            if symbol.startswith(('sh.68', 'sz.30')): return 0.20 # 科创/创业板
            if symbol.startswith('bj'): return 0.30              # 北交所
            return 0.10                                          # 主板
            
        # 应用涨跌停截断 (防止脏数据或除权错误)
        # 我们允许 0.5% 的溢出以容忍数据计算误差
        backtest_df['limit'] = backtest_df['symbol'].apply(get_limit)
        backtest_df.loc[backtest_df['pct_chg_calc'] > backtest_df['limit'] + 0.005, 'pct_chg_calc'] = backtest_df['limit']
        backtest_df.loc[backtest_df['pct_chg_calc'] < -backtest_df['limit'] - 0.005, 'pct_chg_calc'] = -backtest_df['limit']
        
        # 4. 核心改进：交易执行约束（防止“收盘价买入即获利”的未来函数）
        # 逻辑链条：
        # T日：计算因子，产生信号 S_T
        # T+1日：执行日。在收盘判定是否涨跌停。
        #        若可成交，则收盘后持有 S_T。该过程不产生收益。
        # T+2日：持有 S_T 的第一个完整交易日，产生收益 (Close_T+2 / Close_T+1) - 1
        
        # 获取 T 日信号 (相对于当前行的 D 日，信号是 D-1 日产生的)
        backtest_df['signal_T'] = backtest_df.groupby('symbol')['target_weight'].shift(1).fillna(0)
        # 获取 T-1 日的持仓状态 (即在 T+1 执行前的初始状态)
        backtest_df['holding_before'] = backtest_df.groupby('symbol')['actual_holding'].shift(1).fillna(0)
        
        # 判定 T+1 日（当前行）是否可以执行调仓
        def determine_execution(row):
            # 增持/买入：若涨停则失败，维持原仓位
            if row['signal_T'] > row['holding_before'] and row['is_limit_up']:
                return row['holding_before']
            # 减持/卖出：若跌停则失败，维持原仓位
            if row['signal_T'] < row['holding_before'] and row['is_limit_down']:
                return row['holding_before']
            # 正常调仓
            return row['signal_T']
            
        # 计算在 T+1 日收盘后的确切持仓
        backtest_df['actual_holding'] = backtest_df.apply(determine_execution, axis=1)
        
        # 5. 计算组合日收益率 (必须使用前一日收盘后的持仓 * 今日收益率)
        # actual_holding 在 D-1 日结束时确定，决定了 D 日的收益
        backtest_df['holding_for_ret'] = backtest_df.groupby('symbol')['actual_holding'].shift(1).fillna(0)
        backtest_df['stock_ret'] = backtest_df['pct_chg_calc'] * backtest_df['holding_for_ret']
        
        # 6. 计算摩擦成本 (仅在 actual_holding 发生变化时扣除)
        backtest_df['weight_diff'] = backtest_df.groupby('symbol')['actual_holding'].diff().abs().fillna(0)
        daily_turnover = backtest_df.groupby('date')['weight_diff'].sum()
        
        comm = self.config['backtest'].get('commission', 0.00015)
        tax = self.config['backtest'].get('tax', 0.001)
        slippage = self.config['backtest'].get('slippage', 0.0001)
        cost_rate = comm + slippage + (tax * 0.5)
        daily_cost = daily_turnover * cost_rate
        
        portfolio_daily_ret = backtest_df.groupby('date')['stock_ret'].sum() - daily_cost
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
