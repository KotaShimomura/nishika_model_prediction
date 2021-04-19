import pandas as pd  # 基本ライブラリ
import test_predict as tp

df_feature = pd.DataFrame(tp.model.feature_importance(), index=tp.val_x.columns, columns=["importance"]).sort_values(
    "importance",
    ascending=False)

df_feature.to_csv('lgb_shap/featureimportance.csv')
