# 概要

Nishikaの中古マンションデータを用いて、LightgbmとXGboostによる学習・推論を行う。

## 動作検証済み環境
OS: win10  
python: 3.8.5  
docker: 19.03.8

## 注意点
trainデータとtestデータをローカル環境にダウンロードしておく必要がある。  
GPUが使えないためCPUを計算資源として利用している

# 手順

## クローン
```sh
git clone git@github.com:KotaShimomura/nishika_model_prediction.git
```

## 環境構築
```sh
docker/build.sh
```
```sh
docker/run.sh
```

## 学習
```sh
python3 test_predict.py
```

## Lightgbnの特徴量重要度の確認（確認したい場合）
```sh
python3 feature_importance.py
```

## XGboostのshapの確認（確認したい場合）
```sh
python3 xgb_shap.py
```
