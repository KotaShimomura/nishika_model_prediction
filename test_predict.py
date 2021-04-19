import pandas as pd  # 基本ライブラリ
from sklearn.model_selection import train_test_split  # データセット分割用
import glob
import preprocessing_func as pf
from sklearn.metrics import mean_absolute_error as mae

# csvファイルの読み込み
files = glob.glob("train/*.csv")
data_list = []
for file in files:
    data_list.append(pd.read_csv(file, index_col=0))
df = pd.concat(data_list)

# データの前処理
df = pf.data_pre(df)

# データの分割
df_train, df_val = train_test_split(df, test_size=0.2)

# 学習
model = pf.train_model_lgb(df_train, df_val)
# model = pf.train_model_xgb(df_train, df_val)

# 検証用データの準備
col = "取引価格（総額）_log"
val_y = df_val[col]
val_x = df_val.drop(col, axis=1)

# 検証
vals = model.predict(val_x)
print(mae(vals, val_y))

df_test = pd.read_csv("test.csv", index_col=0)
df_test = pf.data_pre(df_test)

predict = model.predict(df_test)
df_test["取引価格（総額）_log"] = predict  # 目的変数に予測結果を代入
df_test[["取引価格（総額）_log"]].to_csv("submit_test.csv")  # 目的変数の部分だけをcsvファイルとして作成

df_feature = pd.DataFrame(model.feature_importance(), index=val_x.columns, columns=["importance"]).sort_values(
    "importance",
    ascending=False)

df_feature.to_csv("featureimportance.csv")
