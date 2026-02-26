import yaml
import logging
import os
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """
    加载 YAML 配置文件
    """
    current_path = Path(__file__).resolve()
    # 假设结构是 src/utils/helpers.py -> 项目根目录需向上三级
    project_root = current_path.parent.parent.parent
    full_path = project_root / config_path
    
    if not full_path.exists():
        full_path = Path(config_path)
        if not full_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {full_path}")
        
    with open(full_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config):
    """
    配置日志系统
    """
    log_cfg = config.get('logging', {})
    level_str = log_cfg.get('level', 'INFO')
    level = getattr(logging, level_str.upper())
    
    # 修正：将相对路径 "." 映射为真实的物理根目录
    project_root = Path(__file__).resolve().parent.parent.parent
    base_dir_val = config['project']['base_dir']
    if base_dir_val == ".":
        base_dir = project_root
    else:
        base_dir = Path(base_dir_val)
        
    log_file = base_dir / log_cfg.get('log_path', 'logs/workflow.log')
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 避免重复添加 handler
    logger = logging.getLogger("QuantAI_Gemini")
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
    return logger

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent
