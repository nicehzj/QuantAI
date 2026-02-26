# 数据库分片说明 (Database Shards)

为了规避 Git 对大文件（如 `.duckdb`）的限制，我们将 `quant_gemini.duckdb` 进行了分片存储。

## 如何还原数据库

在项目根目录下运行以下脚本即可：

```bash
python3 scripts/restore_db.py
```

该脚本将完成以下操作：
1. 合并位于 `database/shards/` 目录下的分片。
2. 解压生成的 `tar.gz` 文件。
3. 还原 `database/quant_gemini.duckdb` 文件。
4. 清理临时的压缩包。

## 注意事项

- 不要将原始的 `.duckdb` 文件提交到 Git（已在 `.gitignore` 中配置）。
- 提交前，请确保分片文件已同步。
