import pandas as pd
import logging
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.utils.helpers import load_config, setup_logging
from src.data.data_loader import DataLoader
from src.factors.factor_engine import FactorEngine
from src.strategy.strategy_base import MultiFactorStrategy
from src.backtest.backtest_engine import BacktestEngine
from src.backtest.report_generator import TextReportGenerator

def run_full_pipeline():
    cfg = load_config()
    logger = setup_logging(cfg)
    logger.info("="*50)
    logger.info("QuantAI Gemini Alpha 评价工作流启动")
    logger.info("="*50)

    # 1. 数据检查与获取
    loader = DataLoader(cfg)
    # 尝试加载 2015 年的股票池 (包含已退市/变动的股票)
    symbols = loader.fetch_historical_universe(date="2015-01-01")[:20] 
    
    # 下载数据 (基准与 20 只历史池股票)
    loader.update_database(symbols) 
    loader.close()

    # 2. 因子计算
    engine = FactorEngine(cfg)
    factor_matrix = engine.get_factor_matrix(start_date="2020-01-01")
    price_df = engine.load_clean_data(start_date="2020-01-01")

    # 3. 策略生成
    strat = MultiFactorStrategy(cfg)
    signals = strat.generate_signals(factor_matrix)
    positions = strat.calculate_positions(signals, target_count=10)
    
    # 4. 回测 (注入基准数据计算 Alpha)
    bt = BacktestEngine(cfg)
    # 重新从数据库读取基准
    from src.data.database import QuantDatabase
    db = QuantDatabase(cfg)
    bench_df = db.read_query("SELECT * FROM benchmark_daily WHERE symbol = 'sh.000300'")
    db.close()
    
    perf, nav = bt.run_vectorized_backtest(positions, price_df, benchmark_df=bench_df)
    
    # 5. 终端报表
    reporter = TextReportGenerator(cfg)
    reporter.display_report(nav, freq='QE')
    
    logger.info("="*50)
    logger.info("策略 Alpha 绩效汇总:")
    for k, v in perf.items():
        logger.info(f"{k}: {v}")
    logger.info("="*50)

if __name__ == "__main__":
    run_full_pipeline()
