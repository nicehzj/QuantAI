import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# 将项目根目录加入 python 搜索路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.utils.helpers import load_config, setup_logging
from src.factors.factor_engine import FactorEngine
from src.optimization.optimizer import StrategyOptimizer
from src.data.database import QuantDatabase

def run_fast_pipeline():
    """
    快速寻优工作流：跳过数据同步，直接进行因子计算与回测
    """
    cfg = load_config()
    logger = setup_logging(cfg)
    
    logger.info("="*60)
    logger.info("QuantAI Gemini 快速寻优管线 (已跳过数据同步)")
    logger.info("="*60)

    # 1. 因子与行情准备 (直接从现有数据库读取)
    engine = FactorEngine(cfg)
    logger.info("正在提取全市场因子矩阵与行情数据 (2020年至今)...")
    factor_matrix = engine.get_factor_matrix(start_date="2020-01-01")
    price_df = engine.load_clean_data(start_date="2020-01-01")
    
    # 2. 加载基准数据 (沪深300)
    db = QuantDatabase(cfg)
    bench_df = db.read_query("SELECT * FROM benchmark_daily WHERE symbol = 'sh.000300'")
    db.close()

    if bench_df.empty:
        logger.warning("未找到基准数据 (sh.000300)，将使用绝对收益评价。")
        bench_df = None

    # 3. 定义寻优空间 (网格寻优范围)
    weight_ranges = {
        'mom_20': [0.3, 0.5],
        'vol_20': [-0.2, -0.1],
        'bias_20': [-0.1, 0.0],
        'vol_ratio': [0.1, 0.2]
    }

    # 4. 执行自动寻优 (基于 T+2/涨跌停/摩擦成本 的最新逻辑)
    optimizer = StrategyOptimizer(cfg)
    best_report, best_factor, _ = optimizer.grid_search_weights(
        factor_matrix, price_df, weight_ranges, benchmark_df=bench_df
    )

    # 5. 使用最佳参数进行最终验证回测
    if best_report:
        from src.strategy.strategy_base import MultiFactorStrategy
        from src.backtest.backtest_engine import BacktestEngine
        
        strat = MultiFactorStrategy(cfg, factor_weights=best_report['weights'])
        signals = strat.generate_signals(factor_matrix)
        positions = strat.calculate_positions(signals, target_count=10)
        
        bt = BacktestEngine(cfg)
        final_perf, nav_series = bt.run_vectorized_backtest(positions, price_df, benchmark_df=bench_df)

        # 6. 输出最终战果报告
        logger.info("\n" + "*"*60)
        logger.info("🏆 快速寻优最终报告 (严格 A 股实操逻辑) 🏆")
        logger.info("*"*60)
        logger.info(f"📍 最佳核心因子: {best_factor}")
        logger.info(f"📍 最优权重配置: {best_report['weights']}")
        logger.info("-" * 40)
        logger.info("📊 核心绩效指标 (已扣除交易费用/摩擦):")
        for k, v in final_perf.items():
            logger.info(f"   - {k}: {v}")
        logger.info("*"*60 + "\n")
    else:
        logger.error("寻优未找到有效结果。")
        
    engine.close()

if __name__ == "__main__":
    run_fast_pipeline()
