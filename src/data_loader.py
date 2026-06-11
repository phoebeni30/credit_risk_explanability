import os
import sys
from pathlib import Path
import pandas as pd
import chardet
import glob

class DataCleaner:
    def __init__(self, data_path: str = "data"):
        self.BASE_PATH = Path(__file__).parent.parent
        self.folder_data_path        = self.BASE_PATH / data_path
        self.raw_folder_data_path    = self._create_dir(self.folder_data_path / "raw")
        self.processed_folder_data_path = self._create_dir(self.folder_data_path / "processed")

 
    def _create_dir(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path
 
    def _read_csv_encoded(self, file_path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path, index_col=False)
        except UnicodeDecodeError:
            raw = open(file_path, "rb").read()
            enc = chardet.detect(raw)["encoding"]
            return pd.read_csv(file_path, encoding=enc, index_col=False, delimiter=",")
 
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [c for c in ["SK_ID_CURR", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"] if c in df.columns]
        df = df.drop(columns=drop_cols)
 
        if "TARGET" in df.columns:
            df = df.rename(columns={"TARGET": "DEFAULT"})
 
        for col in df.select_dtypes(include="object").columns:
            df[col] = pd.Categorical(df[col])
 
        return df
 
 
    def load_raw(self) -> dict[str, pd.DataFrame]:
        files = [f for f in self.raw_folder_data_path.glob("*.csv")
                 if "description" not in f.name]
        print(f"[data]  found {len(files)} raw file(s)")
 
        dfs = {}
        for f in files:
            dfs[f.stem] = self._read_csv_encoded(f)
            print(f"  ← {f.name}  shape={dfs[f.stem].shape}")
        return dfs
 
    def run_cleaner(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dfs = self.load_raw()
 
        # ── train ──
        df_train = self._clean_data(dfs["application_train"])
        print(f"[data]  train shape: {df_train.shape}")
        df_train.to_csv(self.processed_folder_data_path / "data_train.csv", index=False)
 
        # ── test ──
        df_test = self._clean_data(dfs["application_test"])
        print(f"[data]  test  shape: {df_test.shape}")
        df_test.to_csv(self.processed_folder_data_path / "data_test.csv", index=False)
 
        print(f"[data]  saved successfully!")
        return df_train, df_test