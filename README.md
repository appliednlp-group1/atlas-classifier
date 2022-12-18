# Atlas Classifier

## train retriever on rag-japanese

build docker image

```shell
$ docker build -t index:latest -f Dockerfile-index .
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
  -v $(pwd)/data:/data \
  -v $(pwd)/rag-japanese:/app index:latest \
  python3 app/make_small_bert.py \
  --pretrained-model cl-tohoku/bert-base-japanese-whole-word-masking \
  --out-dir data/models/small_bert \
  --num-layers 3
```

preprocess data

```shell
$ docker run -it --rm \
  --gpus all \
  -v $(pwd)/data:/data \
  -v $(pwd)/rag-japanese:/app index:latest \
  python3 app/preprocess_data.py \
  --knowledge-file app/data/knowledge.csv \
  --qa-file app/data/qa.csv \
  --out-file data/dataset/dpr_qa.json \
  --valid-split --out-csv
```

train dpr

```shell
$ docker run -it --rm \
  --gpus all \
  -v $(pwd)/data:/data \
  -v $(pwd)/rag-japanese:/app index:latest \
  python3 app/dpr/train_dense_encoder.py \
  --train_file data/dataset/dpr_qa_train.json \
  --dev_file data/dataset/dpr_qa_valid.json \
  --encoder_model_type hf_bert \
  --pretrained_model_cfg data/models/small_bert \
  --batch_size 8 \
  --output_dir data/models/dpr \
  --num_train_epochs 6
```

convert dpr model to transformers model

```shell
$ docker run -it --rm \
  --gpus all \
  -v $(pwd)/data:/data \
  -v $(pwd)/rag-japanese:/app index:latest \
  python3 app/dpr/convert_model.py \
  -p data/models/dpr/dpr_biencoder.5.386 \
  -o data/models/dpr_transformers
```

convert knowledge to index

```shell
$ docker run -it --rm \
  --gpus all \
  -v $(pwd)/data:/data \
  -v $(pwd)/rag-japanese:/app index:latest \
  python3 app/make_index.py \
  --context-model data/models/dpr_transformers/c_encoder \
  --knowledge-file app/data/knowledge.csv \
  --out-dir data/dataset/dpr_knowledge_index
```

inference by retriever

```shell
$ docker run -it 
```
