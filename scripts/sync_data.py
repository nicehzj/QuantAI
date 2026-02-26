import logging
import sys
from pathlib import Path

# 将项目根目录加入 python 搜索路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.utils.helpers import load_config, setup_logging
from src.data.data_loader import DataLoader

def sync_all_market_data(mode='full'):
    """
    mode: 'full' (全市场同步，不含北交所)
    """
    cfg = load_config()
    logger = setup_logging(cfg)
    
    logger.info("="*60)
    logger.info(f"QuantAI Gemini: 沪深全市场数据同步启动")
    logger.info("="*60)

    try:
        loader = DataLoader(cfg)
        
        # 获取 2015 年至今的沪深全市场快照 (DataLoader 内部已剔除北交所)
        symbols = loader.fetch_historical_universe(date="2015-01-01", universe="all")
        
        logger.info(f"📊 目标同步股票总数: {len(symbols)}")
        
        # 执行增量同步 (force=False，利用断点续传)
        loader.update_database(symbols, batch_size=100, force=False)
        
        logger.info("✅ 沪深全市场数据同步完成！")
        loader.close()
        
    except Exception as e:
        logger.error(f"❌ 数据同步过程中发生异常: {e}", exc_info=True)

if __name__ == "__main__":
    sync_all_market_data()
