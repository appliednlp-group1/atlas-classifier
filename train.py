import torch
from atlas.src.retrievers import Contriever

DEFAULT_RETRIEVER_MODEL_PATH = 'facebook/contriever'


def run(retriever_model_path: str,
        classifier_model_path: str,
        index_datapath: str):
    checkpoint = torch.load(retriever_model_path, map_location='cpu')
    model_dict = {k.replace('retriever.module.contriever.', ''): v
                  for k, v in checkpoint['model']
                  if k.startswith('retriever.module.contriever.')}
    
    retriever = Contriever.from_pretrained(DEFAULT_RETRIEVER_MODEL_PATH)