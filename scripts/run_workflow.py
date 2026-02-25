import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.utils.helpers import load_config, setup_logging
from src.factors.factor_engine import FactorEngine
from src.optimization.optimizer import StrategyOptimizer
from src.backtest.report_generator import TextReportGenerator
from src.data.database import QuantDatabase

def run_pro_pipeline():
    cfg = load_config()
    logger = setup_logging(cfg)
    logger.info("="*60)
    logger.info("QuantAI Gemini 专业级寻优工作流 (全市场模式)")
    logger.info("="*60)

    # 0. 全市场数据同步 (消除存续性偏差)
    from src.data.data_loader import DataLoader
    loader = DataLoader(cfg)
    # 获取 2015 年时的全市场股票快照，包含退市股
    full_symbols = loader.fetch_historical_universe(date="2015-01-01", universe="all")
    logger.info(f"🚀 全市场扫描启动，目标股票数: {len(full_symbols)}")
    # 执行同步 (若本地已有则自动跳过)
    loader.update_database(full_symbols)
    loader.close()

    # 1. 因子与行情准备
    engine = FactorEngine(cfg)
    factor_matrix = engine.get_factor_matrix(start_date="2020-01-01")
    price_df = engine.load_clean_data(start_date="2020-01-01")
    
    db = QuantDatabase(cfg)
    bench_df = db.read_query("SELECT * FROM benchmark_daily WHERE symbol = 'sh.000300'")
    db.close()

    # 2. 定义寻优空间
    weight_ranges = {
        'mom_20': [0.3, 0.5],
        'vol_20': [-0.2],
        'bias_20': [-0.1],
        'vol_ratio': [0.2]
    }

    # 3. 执行自动寻优
    optimizer = StrategyOptimizer(cfg)
    # 我们通过修改 optimizer 结果字典来输出胜率
    best_report, best_factor, _ = optimizer.grid_search_weights(
        factor_matrix, price_df, weight_ranges, benchmark_df=bench_df
    )

    # 再次运行最佳回测以获取完整 perf 字典 (包含胜率)
    from src.strategy.strategy_base import MultiFactorStrategy
    from src.backtest.backtest_engine import BacktestEngine
    strat = MultiFactorStrategy(cfg, factor_weights=best_report['weights'])
    signals = strat.generate_signals(factor_matrix)
    positions = strat.calculate_positions(signals, target_count=10)
    bt = BacktestEngine(cfg)
    final_perf, _ = bt.run_vectorized_backtest(positions, price_df, benchmark_df=bench_df)

    # 4. 输出最终战果报告
    logger.info("\n" + "*"*60)
    logger.info("🏆 全流程自动寻优战果报告 🏆")
    logger.info("*"*60)
    
    if best_report:
        logger.info(f"📍 最佳核心因子: {best_factor}")
        logger.info(f"📍 最优权重配置: {best_report['weights']}")
        logger.info("-" * 40)
        logger.info(f"📊 核心绩效指标:")
        logger.info(f"   - 夏普比率: {final_perf['Sharpe Ratio']}")
        logger.info(f"   - 年化 Alpha: {final_perf.get('Annual Alpha', 'N/A')}")
        logger.info("-" * 40)
        logger.info(f"📈 实战稳健性指标:")
        logger.info(f"   - 策略胜率: {final_perf['Win Rate']}")
        logger.info(f"   - 盈亏比: {final_perf['Win/Loss Ratio']}")
        logger.info(f"   - 总交易笔数: {final_perf['Total Trades']} 笔")
    else:
        logger.error("寻优失败。")
        
    logger.info("*"*60 + "\n")
    engine.close()

if __name__ == "__main__":
    run_pro_pipeline()
