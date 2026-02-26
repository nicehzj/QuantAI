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

def sync_benchmark_data():
    """
    同步基准指数数据 (沪深300)
    """
    cfg = load_config()
    logger = setup_logging(cfg)
    db = QuantDatabase(cfg)
    loader = DataLoader(cfg)
    
    symbol = "sh.000300"
    logger.info(f"开始同步基准指数: {symbol}")
    
    df = loader.download_daily_data(symbol)
    if df is not None and not df.empty:
        # 重命名列以匹配数据库结构 (如果需要)
        # 根据 DataLoader.update_database 的逻辑：
        # final_df.rename(columns={'pctChg': 'pct_chg', 'turn': 'turnover'}, inplace=True)
        df.rename(columns={'pctChg': 'pct_chg', 'turn': 'turnover'}, inplace=True)
        
        # 保存到 benchmark_daily 表
        db.save_dataframe('benchmark_daily', df, if_exists='replace')
        logger.info(f"基准指数 {symbol} 同步完成，共 {len(df)} 条记录。")
    else:
        logger.error(f"未能获取基准指数 {symbol} 的数据。")
    
    loader.close()
    db.close()

if __name__ == "__main__":
    sync_benchmark_data()
