# GEMINI.md - QuantAI Gemini 量化研究工作流指令集

## 1. 项目愿景 (Project Vision)
QuantAI Gemini 是一个专注于**回测真实性**与**超额收益 (Alpha)** 挖掘的专业级量化研究平台。系统通过严格的数据处理逻辑与仿真撮合机制，最大程度消除未来函数与存续性偏差，确保研究成果具备实战参考价值。

## 2. 核心架构与模块职责 (Architecture)

### 2.1 数据中枢 (Data Pipeline)
- **数据源**: 首选 **BaoStock** (稳定、支持后复权)，备选 AkShare。
- **复权方式**: 严格采用 **后复权 (HFQ)** (`adjustflag='1'`)，保持价格序列连续性且无未来数据泄露。
- **存储**: 使用 **DuckDB** 列式数据库，表名 `stock_daily` (行情) 与 `benchmark_daily` (基准)。
- **偏差防控**: 通过 `fetch_historical_universe(date)` 获取历史成分股快照，允许回测中包含已退市股票。

### 2.2 因子与清洗 (Factors & Cleaning)
- **数据清洗**: 严禁使用 `bfill` (向后填充)，仅允许 `ffill` (向前填充)。
- **因子计算**: 采用向量化计算逻辑，涵盖 Momentum, Volatility, Bias 等。
- **预处理**: 每个交易日执行横截面去极值 (Winsorize) 与标准化 (Z-Score)。

### 2.3 高仿真回测 (Backtest Engine)
- **执行延迟**: 采用 **T+2 延迟逻辑** (T日计算信号 -> T+1日收盘成交 -> T+2日起产生收益)，规避零延迟假设。
- **评价体系**: 
  - **绝对指标**: 累积收益、年化收益、最大回撤、夏普比率。
  - **相对指标 (Alpha)**: 年化 Alpha、Beta、信息比率 (IR)。
- **基准**: 默认以沪深300指数 (`sh.000300`) 作为业绩比较基准。

## 3. 运行规范 (Operations)

### 3.1 核心指令
- **全流程运行**: `python3 scripts/run_workflow.py`
- **接口测试**: `python3 scripts/test_data_api.py`
- **模块校验**: `python3 scripts/verify_modules.py`

### 3.2 环境要求
- 依赖库: `pandas`, `baostock`, `duckdb`, `tqdm`, `scipy`, `pyyaml`.
- 终端友好: 报表通过 ASCII 表格打印，无需图形界面。

## 4. 开发约定 (Conventions)
- **中文优先**: 所有文档、注释及交互必须使用中文。
- **进度可见**: 耗时任务 (数据同步、因子处理) 必须集成 `tqdm` 进度条。
- **无偏性原则**: 任何新增加的功能必须审查是否引入“未来函数”或“幸存者偏差”。

## 5. 路线图 (Roadmap)
- [x] 基于 BaoStock 的后复权数据流。
- [x] Alpha/Beta 评价体系。
- [x] 消除存续性偏差的历史股票池逻辑。
- [ ] 接入多核并行因子计算 (Polars 集成)。
- [ ] 实现 Monte Carlo 策略稳健性压力测试。
