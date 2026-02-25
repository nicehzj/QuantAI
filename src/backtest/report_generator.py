import pandas as pd
import numpy as np
import logging

class TextReportGenerator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("QuantAI_Gemini")

    def generate_periodic_report(self, nav_series, freq='ME'):
        """
        生成固定周期的净值报表
        nav_series: index=[date], values=[cumulative_nav]
        freq: 'ME' 为月底, 'QE' 为季度末, 'YE' 为年末
        """
        self.logger.info(f"生成周期性报告 (频率: {freq})...")
        
        # 将 index 转为 DatetimeIndex
        nav_df = nav_series.to_frame(name='nav')
        nav_df.index = pd.to_datetime(nav_df.index)
        
        # 按频率采样 (取周期内最后一天)
        periodic_nav = nav_df.resample(freq).last()
        
        # 计算周期收益率
        periodic_nav['periodic_ret'] = periodic_nav['nav'].pct_change()
        
        # 计算阶段性回撤
        periodic_nav['cum_max'] = periodic_nav['nav'].cummax()
        periodic_nav['drawdown'] = (periodic_nav['nav'] - periodic_nav['cum_max']) / periodic_nav['cum_max']
        
        # 格式化输出表头
        report_lines = []
        report_lines.append("\n" + "="*70)
        report_lines.append(f"{'日期 (Date)':<15} | {'净值 (NAV)':<12} | {'周期收益 (Ret)':<15} | {'最大回撤 (DD)':<15}")
        report_lines.append("-" * 70)
        
        for date, row in periodic_nav.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            nav_val = f"{row['nav']:.4f}"
            ret_val = f"{row['periodic_ret']:.2%}" if not pd.isna(row['periodic_ret']) else "0.00%"
            dd_val = f"{row['drawdown']:.2%}"
            
            report_lines.append(f"{date_str:<15} | {nav_val:<12} | {ret_val:<15} | {dd_val:<15}")
            
        report_lines.append("="*70 + "\n")
        
        report_str = "\n".join(report_lines)
        return report_str

    def display_report(self, nav_series, freq='ME'):
        """
        在终端直接打印报表
        """
        report = self.generate_periodic_report(nav_series, freq)
        print(report)
        return report
