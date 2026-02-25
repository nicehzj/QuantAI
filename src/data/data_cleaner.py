import pandas as pd
import numpy as np
import logging
from scipy.stats import mstats

class DataCleaner:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("QuantAI_Gemini")

    def clean_daily_data(self, df):
        if df is None or df.empty:
            return df
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date'])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        cols_to_fill = ['open', 'close', 'high', 'low', 'volume', 'amount']
        for col in cols_to_fill:
            if col in df.columns:
                df[col] = df.groupby('symbol')[col].ffill()
        return df

    def winsorize_series(self, series, limits=[0.025, 0.025]):
        if series.isnull().all():
            return series
        # 填充缺失值后再去极值
        filled = series.fillna(series.median())
        return pd.Series(mstats.winsorize(filled, limits=limits), index=series.index)

    def standardize_series(self, series):
        std = series.std()
        if std == 0 or pd.isna(std):
            return series - series.mean()
        return (series - series.mean()) / std

    def process_factors(self, factor_df):
        """
        因子横截面预处理 (去极值 & 标准化)
        """
        self.logger.info("开始进行因子横截面预处理 (确保日期对象保留)...")
        
        df = factor_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # 引入进度条
        from tqdm import tqdm
        grouped = df.groupby('date')
        
        results = []
        for date, group in tqdm(grouped, desc="正在处理横截面因子"):
            processed_group = group.copy()
            for col in group.columns:
                if col not in ['date', 'symbol']:
                    # 对每一列进行去极值和标准化
                    processed_group[col] = self.standardize_series(self.winsorize_series(group[col]))
            results.append(processed_group)
            
        return pd.concat(results, ignore_index=True)
