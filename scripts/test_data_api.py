import akshare as ak
import pandas as pd
import logging

def test_akshare():
    print("--- 正在测试 AkShare 接口 ---")
    try:
        # 测试获取股票列表
        print("正在尝试获取股票列表...")
        stock_list = ak.stock_info_a_code_name()
        print(f"成功获取股票列表，样本数量: {len(stock_list)}")
        
        # 测试获取单只股票历史数据
        print("正在尝试获取 000001 历史数据...")
        # akshare 历史行情接口
        df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20240101", end_date="20240110", adjust="qfq")
        if not df.empty:
            print("历史数据获取成功。")
            return True
        else:
            print("获取到的历史数据为空。")
            return False
    except Exception as e:
        print(f"AkShare 运行异常: {e}")
        return False

if __name__ == "__main__":
    ak_ok = test_akshare()
    if not ak_ok:
        print("\nAkShare 测试失败，建议切换至 Baostock。")
    else:
        print("\nAkShare 工作正常。")
