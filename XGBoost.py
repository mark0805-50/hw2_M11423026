import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# 載入 Boston Housing 資料集
boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data.copy()
y = boston.target.copy()

# 移除不是真實特徵
if 'id' in X.columns:
    X = X.drop(columns=['id'])
    print("已移除特徵：id\n")

# 缺值處理
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 設定 K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 不同參數組合
param_sets = [
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50},
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 300},
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 800},
]

# 存放結果
results = []

# 逐一測試不同參數組合
for params in param_sets:
    print(f"\n=== 測試參數組合: {params} ===")
    mape_scores, rmse_scores, r2_scores = [], [], []

    # K-Fold 交叉驗證
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 建立 Pipeline：在每次 fold 內標準化（防止資料洩漏）
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(
                objective="reg:squarederror",
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                **params
            ))
        ])

        # 模型訓練與預測
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # 評估指標
        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mape_scores.append(mape)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print(f"Fold {fold}: MAPE={mape:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    # 計算平均績效
    avg_mape = np.mean(mape_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_r2 = np.mean(r2_scores)

    results.append({
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "n_estimators": params["n_estimators"],
        "MAPE_mean": avg_mape,
        "RMSE_mean": avg_rmse,
        "R2_mean": avg_r2,
    })

    print(f"→ 平均 MAPE={avg_mape:.4f}, 平均 RMSE={avg_rmse:.4f}, 平均 R²={avg_r2:.4f}")

# 輸出平均結果表格
results_df = pd.DataFrame(results)
print("\n=== XGBoost 不同參數組合下的 5-Fold 平均績效 ===")
print(results_df)

# 匯出成 CSV 檔案
output_path = "xgboost_results_pipeline.csv"
results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n 已匯出結果至：{output_path}")
