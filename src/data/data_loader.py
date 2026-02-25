import baostock as bs
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from src.utils.helpers import load_config, setup_logging
from src.data.database import QuantDatabase

class DataLoader:
    def __init__(self, config=None):
        self.config = config if config else load_config()
        self.logger = setup_logging(self.config)
        self.db = QuantDatabase(self.config)
        self.start_date = self.config['data'].get('start_date', '2010-01-01')
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

    def fetch_historical_universe(self, date="2015-01-01"):
        """
        获取历史某一时刻的股票池，以包含后续可能退市的股票，消除存续性偏差
        """
        self.logger.info(f"获取历史股票池 (基准日期: {date})...")
        data_list = []
        # 获取当时的沪深300成分股
        rs = bs.query_hs300_stocks(date=date)
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data()[1])
        
        if not data_list:
            # 兜底：获取当时的全市场列表
            rs = bs.query_all_stock(day=date)
            while (rs.error_code == '0') & rs.next():
                row = rs.get_row_data()
                if row[0].startswith(('sh.', 'sz.')):
                    data_list.append(row[0])
        
        self.logger.info(f"历史股票池获取成功，样本数: {len(data_list)}")
        return data_list

    def download_daily_data(self, symbol):
        """
        下载行情数据 (后复权)，不再进行存续性过滤
        """
        try:
            rs = bs.query_history_k_data_plus(
                symbol,
                "date,open,high,low,close,volume,amount,pctChg,adjustflag,turn",
                start_date=self.start_date,
                end_date=self.last_trading_day,
                frequency="d",
                adjustflag="1" # HFQ
            )
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            if not data_list: return None
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg', 'turn']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df['symbol'] = symbol
            return df
        except Exception as e:
            self.logger.error(f"下载 {symbol} 失败: {e}")
            return None

    def download_benchmark(self, symbol="sh.000300"):
        """
        下载基准指数数据
        """
        self.logger.info(f"正在下载基准数据: {symbol}")
        df = self.download_daily_data(symbol)
        if df is not None:
            self.db.save_dataframe('benchmark_daily', df, if_exists='replace')
        return df

    def update_database(self, symbols=None, batch_size=100):
        if symbols is None:
            symbols = self.fetch_historical_universe()
        
        # 先下载基准
        self.download_benchmark()
        
        total = len(symbols)
        batch_data = []
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="正在同步股票数据")
        
        for i, symbol in enumerate(symbols):
            df = self.download_daily_data(symbol)
            if df is not None: batch_data.append(df)
            if len(batch_data) >= batch_size or i == total - 1:
                if batch_data:
                    final_df = pd.concat(batch_data, ignore_index=True)
                    final_df.rename(columns={'pctChg': 'pct_chg', 'turn': 'turnover'}, inplace=True)
                    self.db.save_dataframe('stock_daily', final_df, if_exists='replace' if i < batch_size else 'append')
                batch_data = []
            pbar.update(1)
        pbar.close()

    def close(self):
        bs.logout()
        self.db.close()
