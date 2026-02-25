import baostock as bs
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from src.utils.helpers import load_config, setup_logging
from src.data.database import QuantDatabase

class DataLoader:
    def __init__(self, config=None):
        self.config = config if config else load_config()
        self.logger = setup_logging(self.config)
        self.db = QuantDatabase(self.config)
        self.start_date = self.config['data'].get('start_date', '2010-01-01')
        self.active_date = "2026-01-01"
        self._login()
        self.last_trading_day = self._get_last_trading_day()

    def _login(self):
        lg = bs.login()
        if lg.error_code != '0':
            self.logger.error(f"BaoStock 登录失败: {lg.error_msg}")
        else:
            self.logger.info("BaoStock 登录成功。")

    def _get_last_trading_day(self):
        today = datetime.now().strftime("%Y-%m-%d")
        start_check = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        rs = bs.query_trade_dates(start_date=start_check, end_date=today)
        trade_dates = []
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            if row[1] == '1':
                trade_dates.append(row[0])
        return trade_dates[-1] if trade_dates else today

    def fetch_historical_universe(self, date="2015-01-01", universe="all"):
        """
        全量股票获取：涵盖沪深主板、创业板、科创板、北交所及已退市股
        """
        self.logger.info(f"正在构建 A 股全市场种子列表 (含北交所)...")
        data_list = []
        
        # 1. 尝试获取基准日期的官方快照
        rs = bs.query_all_stock(day=date)
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            if row[0].startswith(('sh.', 'sz.', 'bj.')):
                data_list.append(row[0])
        
        # 2. 暴力补全：强制扫描所有核心号段，确保 100% 覆盖
        # 即使快照接口漏掉，号段遍历也会将其捕获
        prefixes = [
            ('sh', range(600000, 606000)), # 沪市主板
            ('sz', range(0, 3000)),        # 深市主板/中小板
            ('sz', range(300000, 301500)), # 创业板
            ('sh', range(688000, 689500)), # 科创板
            ('bj', range(830000, 840000)), # 北交所 83段
            ('bj', range(870000, 880000)), # 北交所 87段
            ('bj', range(430000, 440000))  # 北交所 43段
        ]
        
        for prefix, r in prefixes:
            data_list.extend([f"{prefix}.{i:06d}" for i in r])
        
        full_list = sorted(list(set(data_list)))
        self.logger.info(f"全市场种子列表构建完成，总计尝试扫描: {len(full_list)} 个代码。")
        return full_list

    def download_daily_data(self, symbol):
        """
        核心下载函数：获取 2010 年至今的后复权数据
        """
        try:
            rs = bs.query_history_k_data_plus(
                symbol,
                "date,open,high,low,close,volume,amount,pctChg,adjustflag,turn",
                start_date=self.start_date,
                end_date=self.last_trading_day,
                frequency="d",
                adjustflag="1" # 锁定后复权 (HFQ)
            )
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list: return None
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            num_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg', 'turn']
            df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df['symbol'] = symbol
            return df
        except Exception:
            return None

    def update_database(self, symbols=None, batch_size=100):
        if symbols is None:
            symbols = self.fetch_historical_universe()
            
        # 1. 断点续传逻辑：跳过本地已存在的
        existing_stocks = []
        try:
            existing_stocks = self.db.get_all_stocks()
            self.logger.info(f"检测到本地已存有 {len(existing_stocks)} 只股票，准备续传...")
        except:
            pass
            
        todo_symbols = [s for s in symbols if s not in existing_stocks]
        total = len(todo_symbols)
        
        if total == 0:
            self.logger.info("恭喜！所有目标股票已同步，数据库已是最新状态。")
            return

        # 2. 执行分批下载
        batch_data = []
        count = 0
        pbar = tqdm(total=total, desc="全市场数据同步 (含北交所/退市股)")
        
        for i, symbol in enumerate(todo_symbols):
            df = self.download_daily_data(symbol)
            if df is not None and not df.empty:
                batch_data.append(df)
                count += 1
            
            if len(batch_data) >= batch_size or i == total - 1:
                if batch_data:
                    final_df = pd.concat(batch_data, ignore_index=True)
                    final_df.rename(columns={'pctChg': 'pct_chg', 'turn': 'turnover'}, inplace=True)
                    self.db.save_dataframe('stock_daily', final_df, if_exists='append')
                batch_data = []
            
            pbar.update(1)
            
        pbar.close()
        self.logger.info(f"任务结束。本次同步新增: {count} 只有效股票。")

    def close(self):
        bs.logout()
        self.db.close()
