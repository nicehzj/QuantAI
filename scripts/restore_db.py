import os
import subprocess
import glob

def restore_db():
    shard_dir = "database/shards"
    tar_gz_path = "database/quant_gemini.duckdb.tar.gz"
    db_path = "database/quant_gemini.duckdb"
    
    # 获取所有分片文件并按字母顺序排序
    shards = sorted(glob.glob(f"{shard_dir}/db_shard_*"))
    if not shards:
        print("未找到数据库分片文件。")
        return

    print(f"找到 {len(shards)} 个分片，正在合并...")
    
    # 合并分片
    with open(tar_gz_path, "wb") as outfile:
        for shard in shards:
            with open(shard, "rb") as infile:
                outfile.write(infile.read())
    
    print(f"分片已合并到 {tar_gz_path}。正在解压...")
    
    # 解压 tar.gz (包含 quant_gemini.duckdb)
    try:
        # 使用 tar 命令解压到 database 目录
        subprocess.run(["tar", "-xzf", tar_gz_path, "-C", "database"], check=True)
        print(f"数据库已还原到 {db_path}。")
    except Exception as e:
        print(f"解压失败: {e}")
    finally:
        # 删除临时的 tar.gz 文件
        if os.path.exists(tar_gz_path):
            os.remove(tar_gz_path)
            print("临时文件已清理。")

if __name__ == "__main__":
    # 确保在项目根目录下运行
    if os.path.basename(os.getcwd()) != "QuantAI":
        # 尝试切换到 QuantAI 目录
        if os.path.exists("QuantAI"):
            os.chdir("QuantAI")
        else:
            print("请在项目根目录运行此脚本。")
            exit(1)
            
    restore_db()
