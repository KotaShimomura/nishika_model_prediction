import pandas as pd  # 基本ライブラリ
import numpy as np  # 基本ライブラリ
import matplotlib.pyplot as plt  # グラフ描画用
import seaborn as sns
import warnings  # 実行に関係ない警告を無視
import glob
import lightgbm as lgb  # LightGBM
from sklearn.model_selection import train_test_split  # データセット分割用
from sklearn.metrics import mean_absolute_error as mae


def data_pre(df):
    nonnull_list = []
    for col in df.columns:
        nonnull = df[col].count()
        if nonnull == 0:
            nonnull_list.append(col)
    df = df.drop(nonnull_list, axis=1)

    df = df.drop("市区町村名", axis=1)

    df = df.drop("種類", axis=1)

    dis = {
        "30分?60分": 45,
        "1H?1H30": 75,
        "2H?": 120,
        "1H30?2H": 105
    }

    df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].replace(dis).astype(float)

    df["面積（㎡）"] = df["面積（㎡）"].replace("2000㎡以上", 2000).astype(float)

    y_list = {}
    for i in df["建築年"].value_counts().keys():  # 和暦のリスト
        if "平成" in i:
            num = float(i.split("平成")[1].split("年")[0])
            year = 33 - num
        if "令和" in i:
            num = float(i.split("令和")[1].split("年")[0])
            year = 3 - num
        if "昭和" in i:
            num = float(i.split("昭和")[1].split("年")[0])
            year = 96 - num
        y_list[i] = year
    y_list["戦前"] = 76
    df["建築年"] = df["建築年"].replace(y_list)

    year = {
        "年第１四半期": ".25",
        "年第２四半期": ".50",
        "年第３四半期": ".75",
        "年第４四半期": ".99",
    }
    year_list = {}
    for i in df["取引時点"].value_counts().keys():
        for k, j in year.items():  # yearの左がkに右がjに代入され
            if k in i:
                year_rep = i.replace(k, j)  # "年第１四半期"を"0.25"に変換
                year_list[i] = year_rep
    year_list
    df["取引時点"] = df["取引時点"].replace(year_list).astype(float)

    # lgbで使うためにカテゴリカル変数を指定
    for col in ["都道府県名", "地区名", "最寄駅：名称", "間取り", "建物の構造", "用途", "今後の利用目的", "都市計画", "改装", "取引の事情等"]:
        df[col] = df[col].astype("category")

    return df


def train_model_lgb(df_train, df_val):
    # df_train, df_val  = train_test_split(df, test_size = 0.2)

    col = "取引価格（総額）_log"
    train_y = df_train[col]
    train_x = df_train.drop(col, axis=1)

    val_y = df_val[col]
    val_x = df_val.drop(col, axis=1)

    trains = lgb.Dataset(train_x, train_y)
    valids = lgb.Dataset(val_x, val_y)

    params = {
        "objectiv": "regression",
        "metrics": "mae"
    }

    model = lgb.train(params, trains, valid_sets=valids, num_boost_round=1000, early_stopping_rounds=100)
    return model
