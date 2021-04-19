FROM ubuntu:20.04
LABEL maintainer="Kotashimomura"
ENV DEBIAN_FRONTEND noninteractive
ARG INSTALLDIR_PYOCR="/app/ocr"
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y git \
                       make \
                       cmake \
                       gcc \
                       g++ \
                       wget \
                       zip \
                       curl && \
    # ~~~~~Pythonのインストール~~~~~
    apt-get install -y python3 python3-pip && \
    ln -s $(which python3) $(dirname $(which python3))/python  && \
    ln -s $(which pip3) $(dirname $(which python3))/pip && \
    # ~~~~~Python用ML関連ライブラリのインストール~~~~~
    # TensorflowとPytorchは容量が大きいので必要なければコメントアウトしてください。
    # 容量の目安はtensorflow=1.2GB, pytorch=2GBです。
    # Tensorflow, Pytorch以外のML様ライブラリで約2GBです。
    pip install pystache \
                numpy==1.18.5 \
                pandas \
                scikit-learn \
                matplotlib \
                jupyterlab \
                pycaret \
                lightgbm \
                alembic==1.4.1 \
                sqlalchemy==1.3.13 \
                optuna && \
    pip install xgboost && \
    pip install shap && \
    # pip install tensorflow && \
    # pip install torch torchvision && \
    # ~~~~~OpenCVのインストール~~~~~
    # pip install opencv-python && \
    # apt-get install -y libgl1-mesa-dev && \
    # ~~~~Tesseractのインストール~~~~~
    # apt-get install -y libleptonica-dev tesseract-ocr && \
    # ~~~~PyOCRのインストール~~~~~
    # pip install pyocr && \
    # mkdir -p /usr/local/share/tessdata/ && \
    # curl https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/master/jpn.traineddata -sS -L -o /usr/share/tesseract-ocr/4.00/tessdata/jpn.traineddata && \
    # ~~~~MeCabのインストール~~~~
    # apt-get install -y mecab libmecab-dev mecab-ipadic && \
    # pip install --no-binary :all: mecab-python3 && \
    # pip install neologdn && \
    #~~~~作業ディレクトリの作成~~~~
    mkdir -p /home/share

# デフォルトでPythonシェルを起動
#CMD ["python"]