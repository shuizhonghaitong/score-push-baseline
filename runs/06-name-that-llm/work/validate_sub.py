import sys
import pandas as pd

LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']

path = sys.argv[1]
df = pd.read_parquet(path)
assert set(df.columns) >= {"id", "model"}, f"columns: {list(df.columns)}"
assert len(df) == 2699, f"rows {len(df)}"
assert df["id"].is_unique, "ids not unique"
assert not df["model"].isna().any(), "NaN models"
assert df["model"].isin(LABELS).all(), "bad labels"
print("validate OK", path, len(df), "rows; label spread", int(df["model"].nunique()))
