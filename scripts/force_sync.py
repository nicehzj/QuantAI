import baostock as bs
import pandas as pd
import logging
import sys
from pathlib import Path

# 将项目根目录加入 python 搜索路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.utils.helpers import load_config, setup_logging
from src.data.data_loader import DataLoader
from src.data.database import QuantDatabase

def force_sync_core_data():
    cfg = load_config()
    logger = setup_logging(cfg)
    
    db = QuantDatabase(cfg)
    # 彻底删除旧表
    db.conn.execute("DROP TABLE IF EXISTS stock_daily")
    db.close()
    
    loader = DataLoader(cfg)
    # 选取沪深 300 前 50 只作为结构验证
    symbols = loader.fetch_historical_universe(date="2024-01-01")[:50]
    
    logger.info(f"强制同步 {len(symbols)} 只核心股票以更新数据库结构...")
    loader.update_database(symbols, batch_size=10, force=True)
    
    loader.close()
    logger.info("核心数据同步完成。")

if __name__ == "__main__":
    force_sync_core_data()
