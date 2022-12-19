# Atlas Classifier

## cluster上での設定

このリポジトリをホームディレクトリにcloneする (submoduleを忘れないように)

```shell
$ git clone --recursive https://github.com/appliednlp-group1/atlas-classifier.git
```

次に、p系の環境にログインする

```shell
$ srun -p p -t 10:00:00 --gres=gpu:1 --mem=128GB bash
```

次のコマンドは毎回実行する必要がある

```shell
$ export PATH=/home/app/singularity-ce/bin:$PATH
```

次に、ホームディレクトリに自分のsingularityインスタンスを準備する

```shell
$ singularity pull pytorch_22.03-py3.sif docker://nvcr.io/nvidia/pytorch:22.03-py3
```

次に、singularityインスタンスに入る

```shell
$ singularity exec --nv ./<your singularity file name>
```

以下はインスタンス内で行う

まずは環境構築を次のように行う

```shell
$ conda create --name atlas-env python=3.8
$ conda activate atlas-env
$ conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
$ conda install -c pytorch faiss-gpu=1.7.2 cudatoolkit=11.3
$ pip install -r atlas/requirements.txt
```

次に、必要なデータをダウンロードする (それぞれ `~/data` と `~/models` にできる)

```shell
$ cd ~/atlas-classifier
$ python atlas/preprocessing/download_model.py --model models/atlas/base --output_directory ../models
$ python save_dataset.py
```

あとは、train.pyを開始するのみ

```shell
$ python train.py --batch_size 4 --out_dir p101_case1
```


## train retriever on rag-japanese

build docker image

```shell
$ docker build -t index:latest -f docs/Dockerfile-index docs
```

check if gpus are available

```shell
$ docker run -it --rm \
  --gpus all index:latest \
  python3 -c "import torch; print(torch.cuda.is_available())"
```

make bert pretrained model

```shell
$ docker run -it --rm \
  --gpus all \
  -v $(pwd):/app index:latest \
  python3 /app/rag-japanese/make_small_bert.py \
  --pretrained-model cl-tohoku/bert-base-japanese-whole-word-masking \
  --out-dir /app/data/models/small_bert \
  --num-layers 3
```

preprocess data

```shell
$ docker run -it --rm \
  --gpus all \
  -v $(pwd):/app index:latest \
  python3 /app/rag-japanese/preprocess_data.py \
  --knowledge-file /app/rag-japanese/data/knowledge.csv \
  --qa-file /app/rag-japanese/data/qa.csv \
  --out-file /app/data/dataset/dpr_qa.json \
  --valid-split --out-csv
```

train dpr

```shell
$ docker run -it --rm \
  --gpus all \
  -v $(pwd):/app index:latest \
  python3 /app/rag-japanese/dpr/train_dense_encoder.py \
  --train_file /app/data/dataset/dpr_qa_train.json \
  --dev_file /app/data/dataset/dpr_qa_valid.json \
  --encoder_model_type hf_bert \
  --pretrained_model_cfg /app/data/models/small_bert \
  --batch_size 8 \
  --output_dir /app/data/models/dpr \
  --num_train_epochs 6
```

convert dpr model to transformers model

```shell
$ docker run -it --rm \
  --gpus all \
  -v $(pwd):/app index:latest \
  python3 /app/rag-japanese/dpr/convert_model.py \
  -p /app/data/models/dpr/dpr_biencoder.5.386 \
  -o /app/data/models/dpr_transformers
```

convert knowledge to index

```shell
$ docker run -it --rm \
  --gpus all \
  -v $(pwd):/app index:latest \
  python3 /app/rag-japanese/make_index.py \
  --context-model /app/data/models/dpr_transformers/c_encoder \
  --knowledge-file /app/rag-japanese/data/knowledge.csv \
  --out-dir /app/data/dataset/dpr_knowledge_index
```

## infer by retriever on rag-japanese

build docker image

```shell
$ docker build -t atcls:latest -f docs/Dockerfile-atcls docs
```

inference by retriever

```shell
$ docker run -it --rm \
  --gpus all \
  -v $(pwd):/app atcls:latest \
  python3 /app/build_dpretriever.py
```

atlas

```shell
$ docker run -it --rm \
  --gpus all \
  -v $(pwd):/app atcls:latest \
  python3 /app/atlas_classifier.py
```

## ag_news

```shell
$ docker run -it --rm \
  --gpus all \
  -v $(pwd):/app atcls:latest \
  python3 /app/dataloader.py
```


