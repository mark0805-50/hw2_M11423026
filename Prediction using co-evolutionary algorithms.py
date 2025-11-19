# -*- coding: utf-8 -*-
import shutil, os
import time
import numpy as np
import pandas as pd
from pathlib import Path

src_train = r"C:\Users\mark1\OneDrive\Desktop\資料探勘\專案內容\第二次作業\adult\adult.data"
src_test  = r"C:\Users\mark1\OneDrive\Desktop\資料探勘\專案內容\第二次作業\adult\adult.test"
dst_train = os.path.join(os.path.dirname(src_train), "adult.train.txt")
dst_test  = os.path.join(os.path.dirname(src_test),  "adult.test.txt")
shutil.copyfile(src_train, dst_train)
shutil.copyfile(src_test,  dst_test)
print("已建立 adult.train.txt 與 adult.test.txt")

# -*- coding: utf-8 -*-
"""
資料探勘作業：Adult 資料集回歸預測（hours-per-week）
- 訓練：adult.train.txt
- 測試： adult.test.txt
- 目標：預測 hours-per-week
- 模型：KNN、SVR、RandomForest、XGBoost
- 指標：MAPE、MAE、RMSE、R2，並記錄訓練與預測時間
"""

train_path = r"C:\Users\mark1\OneDrive\Desktop\資料探勘\專案內容\第二次作業\adult\adult.train.txt"
test_path  = r"C:\Users\mark1\OneDrive\Desktop\資料探勘\專案內容\第二次作業\adult\adult.test.txt"

# 欄位名稱（UCI Adult 資料集的標準欄位順序）
COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# 讀檔函式：處理 adult.test 的註解行與 income 尾巴的 '.'
def load_adult(filepath: str) -> pd.DataFrame:
    """
    讀取 Adult 檔案為 DataFrame：
    - 跳過以 '|' 開頭的註解行（adult.test 常見）
    - 設定欄位名稱
    - 去除字串值前後空白
    - 將 '?' 視為缺值
    - 將 income 末尾的 '.' 去除（adult.test 常見 ' >50K.' ）
    """
    df = pd.read_csv(
        filepath,
        header=None,               # 檔案通常不含表頭
        names=COLS,
        na_values=["?"],           # 將 '?' 當作缺值
        skipinitialspace=True,     # 去掉逗號後的空白
        comment='|'                # adult.test 前幾行為註解
    )

    # 去除所有字串欄位的前後空白
    obj_cols = df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # income 欄位若有結尾句點，移除（如 '>50K.' -> '>50K'）
    if "income" in df.columns:
        df["income"] = df["income"].str.replace(r"\.$", "", regex=True)

    return df

# 載入資料
train_df = load_adult(train_path)
test_df  = load_adult(test_path)

# 特徵/目標切分
TARGET = "hours-per-week"

# 為避免標籤洩漏，移除目標與 income（income 與工時高度相關，當成特徵會洩漏）
DROP_COLS = [TARGET, "income"]

X_train = train_df.drop(columns=DROP_COLS, errors="ignore")
y_train = train_df[TARGET].astype(float)

X_test  = test_df.drop(columns=DROP_COLS, errors="ignore")
y_test  = test_df[TARGET].astype(float)

# 數值/類別欄位自動辨識
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.columns.difference(num_cols).tolist()

# 前處理器：數值(中位數補值+標準化)、類別(眾數補值+OneHot)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # 數值缺值以中位數補
    ("scaler", StandardScaler())                     # KNN/SVR 受尺度影響，先標準化
])

# 修正點：針對 scikit-learn 新舊版做參數相容
try:
    # sklearn >= 1.2 使用 sparse_output
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    # sklearn < 1.2 使用 sparse
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),   # 類別缺值以眾數補
    ("onehot", ohe)
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ],
    remainder="drop"
)

# 建立四個模型 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# XGBoost 可能尚未安裝，做友善提示
try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError("需要先安裝 xgboost：\n  pip install xgboost") from e

RANDOM_SEED = 42

models = {
    "KNN": KNeighborsRegressor(n_neighbors=10, weights="distance"),
    # SVR 可能偏慢：這裡使用常見參數做基準
    "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_SEED
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist"  # CPU 上較快
    )
}

# 將每個模型包成同一個 Pipeline（同樣的前處理器）
pipelines = {name: Pipeline(steps=[("prep", preprocess), ("model", m)])
             for name, m in models.items()}

# 評估函式：加入 MAPE
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error  # 新增：MAPE 指標
)

def _rmse_compat(y_true, y_pred):
    """相容舊版 sklearn：新版可用 squared=False；舊版改手動開根號。"""
    try:
        # sklearn >= 0.22
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # 舊版沒有 squared 參數
        return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate(model_name: str, pipe: Pipeline, Xtr, ytr, Xte, yte) -> dict:
    """
    進行訓練與預測，回傳各項指標與花費時間。
    """
    # 訓練時間
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0

    # 預測時間
    t1 = time.perf_counter()
    y_pred = pipe.predict(Xte)
    pred_time = time.perf_counter() - t1

    # 新增 MAPE（乘 100 變百分比，例如 7.281）
    mape = mean_absolute_percentage_error(yte, y_pred) * 100
    mae  = mean_absolute_error(yte, y_pred)
    rmse = _rmse_compat(yte, y_pred)
    r2   = r2_score(yte, y_pred)

    return {
        "Model": model_name,
        "MAPE": round(mape, 3),
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3),
        "Train_Time(s)": round(train_time, 3),
        "Predict_Time(s)": round(pred_time, 3)
    }

# 逐一評估、彙整結果
results = []
for name, pipe in pipelines.items():
    print(f"訓練與評估：{name} ...")
    res = evaluate(name, pipe, X_train, y_train, X_test, y_test)
    results.append(res)

results_df = pd.DataFrame(results).sort_values(by="RMSE")  # 以 RMSE 由小到大排序
print("\n=== 模型表現（以 RMSE 排序） ===")
print(results_df.to_string(index=False))

# 輸出成 CSV 方便繳交或附錄
out_path = Path.cwd() / "adult_regression_results.csv"
results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n已輸出結果至：{out_path.resolve()}")