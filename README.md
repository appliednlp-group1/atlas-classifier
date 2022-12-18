# Atlas Classifier

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
