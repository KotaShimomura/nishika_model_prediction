import lightgbm as lgb  # LightGBM
import xgboost as xgb


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

    model = lgb.train(
        params,
        trains,
        valid_sets=valids,
        num_boost_round=500,
        early_stopping_rounds=10,
        verbose_eval=10
    )

    return model


def train_model_xgb(df_train, df_val):
    col = "取引価格（総額）_log"
    train_y = df_train[col]
    train_x = df_train.drop(col, axis=1)

    val_y = df_val[col]
    val_x = df_val.drop(col, axis=1)

    trains = xgb.DMatrix(train_x, label=train_y)
    # 検証用
    valids = xgb.DMatrix(val_x, label=val_y)

    params = {
        "objectiv": "regression",
        "metrics": "mae"
    }

    results_dict = {}

    model = xgb.train(
        params,
        trains,
        evals=[(trains, "train"), (valids, "valid")],
        num_boost_round=500,
        early_stopping_rounds=10,
        verbose_eval=10,
        evals_result=results_dict,

    )

    return model
