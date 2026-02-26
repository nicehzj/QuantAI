import duckdb
import pandas as pd
from pathlib import Path
import logging

class QuantDatabase:
    def __init__(self, config):
        """
        初始化 DuckDB 数据库连接
        """
        self.config = config
        
        # 修正路径解析逻辑
        project_root = Path(__file__).resolve().parent.parent.parent
        base_dir_val = config['project']['base_dir']
        if base_dir_val == ".":
            base_dir = project_root
        else:
            base_dir = Path(base_dir_val)
            
        db_path = base_dir / config['database'].get('db_path', 'database/quant_gemini.duckdb')
        
        # 确保数据库所在目录存在
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = str(db_path)
        self.logger = logging.getLogger("QuantAI_Gemini")
        
        # 初始化数据库连接
        self.conn = duckdb.connect(self.db_path)
        self.logger.info(f"DuckDB 数据库已连接: {self.db_path}")

    def save_dataframe(self, table_name, df, if_exists='append'):
        """
        将 DataFrame 存入 DuckDB
        """
        if df.empty:
            return
        
        # DuckDB 可以直接将 pandas df 注册为临时表并插入
        if if_exists == 'replace':
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        else:
            # 检查表是否存在
            table_exists = self.conn.execute(
                f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
            ).fetchone()[0] > 0
            
            if not table_exists:
                self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            else:
                self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        
        self.logger.info(f"数据已保存至表: {table_name}, 记录数: {len(df)}")

    def read_query(self, query):
        """
        执行查询并返回 DataFrame
        """
        return self.conn.execute(query).df()

    def get_all_stocks(self, table_name='stock_daily'):
        """
        获取库中所有股票代码
        """
        return self.conn.execute(f"SELECT DISTINCT symbol FROM {table_name}").df()['symbol'].tolist()

    def close(self):
        """
        关闭数据库连接
        """
        self.conn.close()
        self.logger.info("数据库连接已关闭")
