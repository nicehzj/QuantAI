import pandas as pd
import numpy as np
import os
import sys

# 导入配置加载器
from src.utils.helpers import load_config

def verify_backtest_bug():
    print("--- 深度解析 BacktestEngine.py 中的核心 Bug ---")
    
    # 场景：股票 A 在 2020-01-01 暴涨，2020-02-01 也暴涨。
    # 它只在这两天进入了选股前 10 名 (positions)
    
    positions_data = {
        'date': pd.to_datetime(['2020-01-01', '2020-02-01']), 
        'symbol': ['A', 'A'],
        'target_weight': [0.1, 0.1]
    }
    positions = pd.DataFrame(positions_data)
    
    # Bug 点：在稀疏 DataFrame 上直接 shift
    positions = positions.sort_values(['symbol', 'date'])
    positions['actual_weight_bug'] = positions.groupby('symbol')['target_weight'].shift(2)
    
    print("【错误实现】稀疏矩阵 shift(2):")
    print(positions)
    print("由于 2020-01-01 是第一条记录，shift(2) 之后它会在 2020-02-01 看到 NaN (如果只有两条数据的话)。")
    print("但如果股票 A 在 2019 年也有数据，那么 2020-01-01 会承接 2019 年末的信号，逻辑完全混乱。")

    # 模拟真实时间轴
    dates = pd.date_range('2020-01-01', '2020-02-10')
    price_df = pd.DataFrame({'date': dates, 'symbol': 'A', 'close': np.random.rand(len(dates))})
    
    # 正确的做法应该是：先 reindex 到全时间轴，再 shift
    full_positions = positions[['date', 'symbol', 'target_weight']].set_index(['date', 'symbol'])
    # 重新索引到所有交易日
    idx = pd.MultiIndex.from_product([dates, ['A']], names=['date', 'symbol'])
    correct_positions = full_positions.reindex(idx).fillna(0)
    correct_positions['actual_weight_correct'] = correct_positions.groupby('symbol')['target_weight'].shift(2)
    
    print("\n【正确实现】全时间轴 shift(2):")
    print(correct_positions.loc[('2020-01-01'):('2020-01-05')])
    print("信号会在 T+2 日（即 2020-01-03）准确生效，且不会错位。")

def check_db_anomalies():
    print("\n--- 检查数据库异常收益 ---")
    import duckdb
    config = load_config()
    db_path = config['database'].get('db_path', 'database/quant_gemini.duckdb')
    if not os.path.exists(db_path):
        print(f"数据库不存在: {db_path}")
        return

    conn = duckdb.connect(db_path)
    # 检查是否存在复权导致的巨幅涨跌 (例如 100% 以上的涨幅或 -90% 的跌幅)
    query = """
    SELECT symbol, date, close, pct_chg 
    FROM stock_daily 
    WHERE pct_chg > 21 OR pct_chg < -21 
    LIMIT 10
    """
    res = conn.execute(query).df()
    if not res.empty:
        print("警告：数据库中发现异常波动记录（单日涨跌超 21%）:")
        print(res)
    else:
        print("数据库日内波动在正常范围内 (未见 21% 以上的异常信号)。")
    conn.close()

if __name__ == "__main__":
    verify_backtest_bug()
    try:
        check_db_anomalies()
    except Exception as e:
        print(f"数据库检查失败: {e}")
