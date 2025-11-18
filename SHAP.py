import sys
import pandas as pd
import numpy as np
import shap
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import xgboost
from sklearn.datasets import fetch_openml

# 顯示版本與環境 
print("========== 環境檢查 ==========")
print(f"Python 路徑：{sys.executable}")
print(f"xgboost 版本：{xgboost.__version__}")
print(f"xgboost 位置：{xgboost.__file__}")
print(f"shap 版本：{shap.__version__}")
print(f"shap 位置：{shap.__file__}")
print("=================================\n")

# 讀取資料（改為內建 Boston Housing dataset）
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame
print(f"內建 Boston Housing dataset 已載入，共 {df.shape[0]} 筆資料、{df.shape[1]} 欄位。")

# 處理缺值
imputer = SimpleImputer(strategy='median')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 分離特徵與目標
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# 移除不是真實特徵
if 'id' in X.columns:
    X = X.drop(columns=['id'])
    print("已移除特徵：id\n")

# XGBoost 模型設定
model = XGBRegressor(random_state=42, n_estimators=300, learning_rate=0.1,max_depth=4,subsample=0.8,colsample_bytree=0.8,)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 使用所有特徵先測一次
mape_list, rmse_list, r2_list = [], [], []
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mape_list.append(mean_absolute_percentage_error(y_test, y_pred))
    rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2_list.append(r2_score(y_test, y_pred))

print("===== 使用全部特徵的平均績效 =====")
print(f"平均 MAPE = {np.mean(mape_list):.4f}")
print(f"平均 RMSE = {np.mean(rmse_list):.4f}")
print(f"平均 R² = {np.mean(r2_list):.4f}")

# SHAP 特徵重要性
print("\n===== 使用 SHAP 計算特徵重要性 =====")
model.fit(X, y)
explainer = shap.Explainer(model)
shap_values = explainer(X)

feature_importance = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'SHAP_Importance': feature_importance
}).sort_values(by='SHAP_Importance', ascending=False)

print("\n特徵重要性排名：")
print(importance_df)

# 測試多個 SHAP 門檻 
thresholds = [0.6, 0.4, 0.2]
results = []

for t in thresholds:
    selected_features = importance_df[importance_df['SHAP_Importance'] > t]['Feature']
    X_selected = X[selected_features]

    mape_list_t, rmse_list_t, r2_list_t = [], [], []
    for train_idx, test_idx in kf.split(X_selected):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mape_list_t.append(mean_absolute_percentage_error(y_test, y_pred))
        rmse_list_t.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_list_t.append(r2_score(y_test, y_pred))

    results.append({
        "門檻": t,
        "特徵數": len(selected_features),
        "MAPE": np.mean(mape_list_t),
        "RMSE": np.mean(rmse_list_t),
        "R2": np.mean(r2_list_t),
        "保留特徵": list(selected_features)
    })

# 顯示結果
print("\n===== 各 SHAP 門檻績效比較 =====")
for res in results:
    print(f"\n--- 門檻 > {res['門檻']} ---")
    print(f"保留特徵數：{res['特徵數']}")
    print(f"平均 MAPE = {res['MAPE']:.4f}")
    print(f"平均 RMSE = {res['RMSE']:.4f}")
    print(f"平均 R² = {res['R2']:.4f}")
    print(f"保留特徵：{res['保留特徵']}")

# 儲存結果表格
results_df = pd.DataFrame(results)
results_df = results_df.drop(columns=["保留特徵"])
results_df.to_csv("shap_threshold_comparison.csv", index=False)
print("\n 各門檻結果已輸出至 shap_threshold_comparison.csv")
