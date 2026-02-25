import pandas as pd
import numpy as np
import logging
from src.data.database import QuantDatabase
from src.data.data_cleaner import DataCleaner

class FactorEngine:
    def __init__(self, config):
        """
        初始化因子计算引擎
        """
        self.config = config
        self.logger = logging.getLogger("QuantAI_Gemini")
        self.db = QuantDatabase(self.config)
        self.cleaner = DataCleaner(self.config)

    def load_clean_data(self, start_date=None, end_date=None):
        """
        从数据库加载清洗后的行情数据
        """
        query = "SELECT * FROM stock_daily"
        if start_date:
            query += f" WHERE date >= '{start_date}'"
        if end_date:
            query += f" AND date <= '{end_date}'"
        
        df = self.db.read_query(query)
        self.logger.info(f"数据加载完成，共计 {len(df)} 条记录。")
        
        # 基础数据清洗
        df = self.cleaner.clean_daily_data(df)
        return df

    def calculate_technical_factors(self, df):
        """
        计算基础技术因子 (向量化)
        """
        self.logger.info("开始计算基础技术因子...")
        df = df.sort_values(['symbol', 'date'])
        
        # 1. 动量因子 (5日/20日收益率)
        df['mom_5'] = df.groupby('symbol')['close'].pct_change(periods=5)
        df['mom_20'] = df.groupby('symbol')['close'].pct_change(periods=20)
        
        # 2. 波动率因子 (20日收盘价波动率)
        df['vol_20'] = df.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(20).std())
        
        # 3. 均线乖离度 (Bias)
        df['ma_20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
        df['bias_20'] = (df['close'] - df['ma_20']) / df['ma_20']
        
        # 4. 成交量因子 (5日/20日成交额比值)
        df['vol_ma_5'] = df.groupby('symbol')['amount'].transform(lambda x: x.rolling(5).mean())
        df['vol_ma_20'] = df.groupby('symbol')['amount'].transform(lambda x: x.rolling(20).mean())
        df['vol_ratio'] = df['vol_ma_5'] / df['vol_ma_20']
        
        return df

    def get_factor_matrix(self, start_date=None, end_date=None, apply_preprocessing=True):
        """
        获取因子矩阵并进行预处理 (横截面去极值与标准化)
        """
        raw_df = self.load_clean_data(start_date, end_date)
        factor_df = self.calculate_technical_factors(raw_df)
        
        # 筛选有效因子列 (mom_5, mom_20, vol_20, bias_20, vol_ratio)
        factor_cols = ['date', 'symbol', 'mom_5', 'mom_20', 'vol_20', 'bias_20', 'vol_ratio']
        matrix = factor_df[factor_cols].dropna()
        
        if apply_preprocessing:
            # 执行横截面标准化处理
            matrix = self.cleaner.process_factors(matrix)
        
        # 强制重置索引，确保 date 和 symbol 都在列中
        if 'date' not in matrix.columns:
            matrix = matrix.reset_index()
            # 如果 reset_index 产生了 'index' 列，重命名为 'date'
            if 'index' in matrix.columns:
                matrix = matrix.rename(columns={'index': 'date'})
            
        self.logger.info(f"最终因子矩阵列名: {matrix.columns.tolist()}")
        return matrix

    def close(self):
        self.db.close()

if __name__ == "__main__":
    from src.utils.helpers import load_config, setup_logging
    cfg = load_config()
    setup_logging(cfg)
    
    engine = FactorEngine(cfg)
    # factor_matrix = engine.get_factor_matrix(start_date="2020-01-01")
    # print(factor_matrix.head())
    engine.close()
