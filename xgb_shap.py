import pandas as pd  # 基本ライブラリ
import numpy as np  # 基本ライブラリ
import matplotlib.pyplot as plt  # グラフ描画用
from matplotlib.backends.backend_pdf import PdfPages
import model as model
import test_predict as tp
import shap

shap.initjs()

explainer = shap.TreeExplainer(
    model=tp.model,
    model_output='margin'
)

explainer = shap.TreeExplainer(tp.model)

shap_values = explainer.shap_values(X=tp.val_x)

shap.summary_plot(shap_values, tp.df_test, show=False)
plt.title('summary_plot')
plt.savefig('lgb_shap/summary_plot.png', bbox_inches='tight')

shap.summary_plot(shap_values, tp.df_test, show=False)
pp = PdfPages('lgb_shap/summary_plot.pdf')
plt.title('summary_plot')
fig = plt.gcf()
pp.savefig(fig, bbox_inches='tight')
pp.close()
