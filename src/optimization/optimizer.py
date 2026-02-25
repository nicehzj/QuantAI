import pandas as pd
import numpy as np
import logging
from scipy.optimize import minimize
from src.strategy.strategy_base import MultiFactorStrategy
from src.backtest.backtest_engine import BacktestEngine

class StrategyOptimizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("QuantAI_Gemini")
        self.engine = BacktestEngine(self.config)

    def grid_search_weights(self, factor_matrix, price_df, weight_ranges):
        """
        因子权重网格搜索寻优
        """
        self.logger.info("启动网格搜索进行权重寻优...")
        import itertools
        factors = list(weight_ranges.keys())
        weight_combinations = list(itertools.product(*weight_ranges.values()))
        
        best_sharpe = -np.inf
        best_params = None
        results = []

        for weights in weight_combinations:
            current_weights = dict(zip(factors, weights))
            strat = MultiFactorStrategy(self.config, factor_weights=current_weights)
            signals = strat.generate_signals(factor_matrix)
            positions = strat.calculate_positions(signals, target_count=10)
            
            perf, _ = self.engine.run_vectorized_backtest(positions, price_df)
            sharpe = float(perf['Sharpe Ratio'].replace(' (N/A)', '0')) if 'N/A' not in perf['Sharpe Ratio'] else 0
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = current_weights
                
        self.logger.info(f"寻优完成。最佳权重: {best_params}")
        return best_params

    # --- 资产组合优化算法 ---
    
    def calculate_risk_parity_weights(self, returns_df):
        """
        计算风险平价权重 (Risk Parity)
        returns_df: 各资产(股票)的历史收益率序列
        目标: 使各资产对组合风险的贡献相等
        """
        cov = returns_df.cov().values
        num_assets = len(returns_df.columns)
        
        def risk_budget_objective(weights, cov):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            # 计算各资产风险贡献 (Marginal Risk Contribution)
            mrc = np.dot(cov, weights) / portfolio_vol
            risk_contribution = weights * mrc
            # 目标: 最小化风险贡献之间的方差
            target_risk = portfolio_vol / num_assets
            return np.sum(np.square(risk_contribution - target_risk))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_weights = [1.0 / num_assets] * num_assets
        
        res = minimize(risk_budget_objective, init_weights, args=(cov,),
                       method='SLSQP', bounds=bounds, constraints=constraints)
        
        return pd.Series(res.x, index=returns_df.columns)

    def calculate_mvo_weights(self, returns_df, target_return=None):
        """
        Markowitz 均值-方差优化 (Mean-Variance Optimization)
        目标: 在给定收益下最小化风险，或最大化夏普比率
        """
        mean_ret = returns_df.mean()
        cov = returns_df.cov()
        num_assets = len(returns_df.columns)

        def objective(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            port_ret = np.dot(weights, mean_ret)
            # 最大化夏普比率 (简略版，无风险收益设为0)
            return -port_ret / port_vol if port_vol != 0 else 0

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_weights = [1.0 / num_assets] * num_assets
        
        res = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return pd.Series(res.x, index=returns_df.columns)
