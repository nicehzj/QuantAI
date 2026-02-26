from pathlib import Path
import yaml
import sys

def test_paths():
    project_root = Path("/home/Admin/QuantAI")
    config_path = project_root / "config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_dir_val = config['project']['base_dir']
    if base_dir_val == ".":
        base_dir = project_root
    else:
        base_dir = Path(base_dir_val)
    
    db_path = base_dir / config['database'].get('db_path', 'database/quant_gemini.duckdb')
    print(f"Base Dir: {base_dir}")
    print(f"DB Path: {db_path}")

if __name__ == "__main__":
    test_paths()
