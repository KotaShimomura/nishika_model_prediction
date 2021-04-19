import pandas as pd  # 基本ライブラリ
import numpy as np  # 基本ライブラリ
import matplotlib.pyplot as plt  # グラフ描画用
import model as model
import test_predict as tp
import shap

shap.initjs()

explainer = shap.TreeExplainer(
    model=tp.model,
    model_output='margin'
)

# shap_values = explainer.shap_values(X=tp.df_train)
shap_values = explainer.shap_values(X=tp.df_train)

shap.summary_plot(shap_values, tp.df_train, plot_type="bar")

shap.summary_plot(shap_values, tp.df_train)

shap.force_plot(base_value=explainer.expected_value, shap_values=shap_values, features=tp.df_train)

shap.dependence_plot(ind="RM", shap_values=shap_values, features=tp.df_train)
