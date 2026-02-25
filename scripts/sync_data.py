import logging
import sys
from pathlib import Path

# 将项目根目录加入 python 搜索路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.utils.helpers import load_config, setup_logging
from src.data.data_loader import DataLoader

def sync_all_market_data():
    """
    仅执行 A 股全市场数据下载与同步
    """
    cfg = load_config()
    logger = setup_logging(cfg)
    
    logger.info("="*60)
    logger.info("QuantAI Gemini: 全市场数据同步任务启动")
    logger.info("="*60)

    try:
        loader = DataLoader(cfg)
        
        # 1. 获取全历史股票快照 (2015-01-01 当时在市的所有股票，含退市股)
        # 这样可以彻底消除存续性偏差
        full_symbols = loader.fetch_historical_universe(date="2015-01-01", universe="all")
        
        logger.info(f"📊 目标同步股票总数: {len(full_symbols)}")
        logger.info("数据将以'后复权 (HFQ)'标准存入 DuckDB 本地数据库...")
        
        # 2. 开始执行分批同步 (内部已集成 tqdm 进度条)
        # 注意：全市场抓取建议分批运行，如果中断，再次运行会自动跳过已存在的
        loader.update_database(full_symbols, batch_size=100)
        
        logger.info("="*60)
        logger.info("✅ 全市场数据同步任务圆满完成！")
        logger.info("="*60)
        
        loader.close()
        
    except Exception as e:
        logger.error(f"❌ 数据同步过程中发生异常: {e}", exc_info=True)

if __name__ == "__main__":
    sync_all_market_data()
