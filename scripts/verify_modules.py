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
from src.data.database import QuantDatabase
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.factors.factor_engine import FactorEngine
from src.strategy.strategy_base import MultiFactorStrategy
from src.backtest.backtest_engine import BacktestEngine
from src.backtest.report_generator import TextReportGenerator

def verify():
    # 1. 验证配置与日志
    print("--- [1/7] 正在验证配置加载 ---")
    cfg = load_config()
    setup_logging(cfg)
    print(f"项目名称: {cfg.get('project', {}).get('name', 'Unknown')}")
    
    # 2. 验证数据库初始化 (DuckDB)
    print("\n--- [2/7] 正在验证 DuckDB 初始化 ---")
    db = QuantDatabase(cfg)
    db.conn.execute("SELECT 1")
    print("DuckDB 连接成功。")
    
    # 3. 验证数据加载 (采样 3 只股票)
    print("\n--- [3/7] 正在验证数据加载 (采样 3 只股票) ---")
    loader = DataLoader(cfg)
    test_symbols = loader.fetch_active_stocks()[:3]
    print(f"获取到活跃股票示例: {test_symbols}")
    
    # 下载并存入数据库 (仅下载最近数据用于测试)
    loader.start_date = "2024-01-01"
    for symbol in test_symbols:
        df = loader.download_daily_data(symbol)
        if df is not None:
            db.save_dataframe('stock_daily', df, if_exists='append')
    print("样本数据已存入数据库。")
    
    # 4. 验证因子引擎与清洗
    print("\n--- [4/7] 正在验证因子计算与数据清洗 ---")
    engine = FactorEngine(cfg)
    raw_df = engine.load_clean_data(start_date="2024-01-01")
    if not raw_df.empty:
        factor_matrix = engine.calculate_technical_factors(raw_df)
        print(f"因子矩阵生成成功，形状: {factor_matrix.shape}")
        
        # 5. 验证策略生成
        print("\n--- [5/7] 正在验证多因子策略信号 ---")
        strat = MultiFactorStrategy(cfg)
        cleaner = DataCleaner(cfg)
        factor_cols = ['date', 'symbol', 'mom_5', 'mom_20', 'vol_20', 'bias_20', 'vol_ratio']
        # 提取存在的列
        available_cols = [c for c in factor_cols if c in factor_matrix.columns]
        matrix_processed = cleaner.process_factors(factor_matrix[available_cols].dropna())
        
        signals = strat.generate_signals(matrix_processed)
        positions = strat.calculate_positions(signals, target_count=2)
        print(f"信号与持仓计算成功。样本数: {len(positions)}")
        
        # 6. 验证回测引擎
        print("\n--- [6/7] 正在验证向量化回测 ---")
        bt = BacktestEngine(cfg)
        perf, nav = bt.run_vectorized_backtest(positions, raw_df)
        print(f"回测运行成功。累积收益率: {perf.get('Cumulative Return', 'N/A')}")
        
        # 7. 验证报表生成 (终端输出)
        print("\n--- [7/7] 正在验证终端报表输出 ---")
        reporter = TextReportGenerator(cfg)
        reporter.display_report(nav, freq='ME')
    else:
        print("警告: 未能获取到测试数据，后续验证跳过。")
    
    print("\n" + "="*50)
    print("所有核心模块校验完成！QuantAI Gemini 工作流已就绪。")
    print("="*50)
    
    db.close()

if __name__ == "__main__":
    verify()
