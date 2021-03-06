import pandas as pd  # 基本ライブラリ
from sklearn.model_selection import train_test_split  # データセット分割用
import glob
import preprocessing_func as pf
import model as model
import xgboost as xgb
from sklearn.metrics import mean_absolute_error as mae
import time

# csvファイルの読み込み
print("csvファイルを読み込むペコ")
files = glob.glob("train/*.csv")
data_list = []
for file in files:
    data_list.append(pd.read_csv(file, index_col=0))
df = pd.concat(data_list)
df_test = pd.read_csv("test.csv", index_col=0)
print("読み込み完了ペコ")

df_list = df

# lgbデータの前処理
df = pf.data_pre(df)
# xgbデータの前処理
# df = pf.data_pre1(df)

# データの分割
df_train, df_val = train_test_split(df, test_size=0.2)

# modelの学習にかかる時間を計測する
model_study_time_start = time.time()

# 学習
print("モデルの学習を始めるペコ")
model = model.train_model_lgb(df_train, df_val)
print("モデルの学習が終わったペコ")
# model = model.train_model_xgb(df_train, df_val)

# 学習時間の表示
model_study_time_finish = time.time()
elapsed_time = model_study_time_finish - model_study_time_start
print(f"モデルの学習にかかった時間は : {elapsed_time}ぺこ")

# 検証用データの準備
col = "取引価格（総額）_log"
val_y = df_val[col]
val_x = df_val.drop(col, axis=1)

print("検証データでモデルの評価をするペコ")
# 検証lgb
vals = model.predict(val_x)
# 検証xgb
# vals = model.predict(xgb.DMatrix(val_x))
print(mae(vals, val_y))

# テストデータの加工lgb
df_test = pf.data_pre(df_test)
# テストデータの加工xgb
# df_test = pf.data_pre1(df_test)

print("テストデータでモデルの評価をするペコ")
# lgb
predict = model.predict(df_test)
# xgb
# predict = model.predict(xgb.DMatrix(df_test))

print("結果をcsvファイルで保存するペコ")
df_test["取引価格（総額）_log"] = predict  # 目的変数に予測結果を代入
df_test[["取引価格（総額）_log"]].to_csv("submit_test.csv")  # 目的変数の部分だけをcsvファイルとして作成
