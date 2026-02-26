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
from src.data.database import QuantDatabase

def run_adaptive_pipeline():
    """
    全自适应专家管线：大盘环境感知 + 动态因子权重 + 专家级止损
    """
    cfg = load_config()
    logger = setup_logging(cfg)
    
    logger.info("="*60)
    logger.info("QuantAI Gemini 全自适应专家管线")
    logger.info("环境识别: Bull/Bear/Sideways | 因子: 趋势/反转/止损")
    logger.info("="*60)

    # 1. 因子与行情准备 (全量因子矩阵)
    engine = FactorEngine(cfg)
    logger.info("正在提取全市场因子矩阵与行情数据 (2020年至今)...")
    factor_matrix = engine.get_factor_matrix(start_date="2020-01-01")
    price_df = engine.load_clean_data(start_date="2020-01-01")
    
    # 2. 加载基准数据
    db = QuantDatabase(cfg)
    bench_df = db.read_query("SELECT * FROM benchmark_daily WHERE symbol = 'sh.000300'")
    db.close()

    # 3. 执行自适应策略
    from src.strategy.strategy_base import MultiFactorStrategy
    from src.backtest.backtest_engine import BacktestEngine
    
    strat = MultiFactorStrategy(cfg)
    # 核心：策略类内部已实现 Regime Detection 和动态调仓
    positions = strat.calculate_positions(
        factor_matrix, 
        price_df, 
        target_count=20, 
        benchmark_df=bench_df,
        stop_loss_pct=0.08,
        min_hold_days=10  # 强制锁仓 10 个交易日
    )
    
    # 4. 运行回测
    bt = BacktestEngine(cfg)
    final_perf, nav_series = bt.run_vectorized_backtest(positions, price_df, benchmark_df=bench_df)

    # 5. 输出最终战果报告
    logger.info("\n" + "*"*60)
    logger.info("🏆 全自适应策略回测报告 🏆")
    logger.info("*"*60)
    for k, v in final_perf.items():
        logger.info(f"   - {k}: {v}")
    logger.info("*"*60 + "\n")
    
    engine.close()

if __name__ == "__main__":
    run_adaptive_pipeline()
