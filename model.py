import lightgbm as lgb  # LightGBM
import xgboost as xgb
import matplotlib.pyplot as plt  # グラフ描画用
from matplotlib.backends.backend_pdf import PdfPages
import shap


def train_model_lgb(df_train, df_val):

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
        num_boost_round=100,
        early_stopping_rounds=10,
        verbose_eval=10
    )

    model.params["objective"] = "regression"

    shap.initjs()

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X=train_x)

    shap.summary_plot(shap_values, train_x, plot_type="bar")
    plt.title('summary_plot')
    plt.savefig('lgb_shap/summary_plot.png', bbox_inches='tight')

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
        num_boost_round=100,
        early_stopping_rounds=10,
        verbose_eval=10,
        evals_result=results_dict,

    )

    return model
