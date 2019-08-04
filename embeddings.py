from pytorch_pretrained_bert import BertTokenizer, BertModel
# import tensorflow as tf
# import tensorflow_hub as hub
import torch
import numpy as np


def load_bert_model(cuda=False):
    model = BertModel.from_pretrained('bert-base-uncased')
    if cuda:
        model.to('cuda')
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return (model, tokenizer)


def generate_vecs_bert(models, document, type='vector', cuda=False):
    if type not in ['vector', 'matrix']:
        raise ValueError('type must be vector or matrix')

    model, tokenizer = models

    print('[INFO] BERT encoder starts working')
    embedding = list()
    len_doc = len(document)
    for n, sent in enumerate(document):
        # print the status
        # if n % 100 == 0:
        #     print('### PROCESSING {} out of {}'.format(n, len_doc))
        print(n)

        tokenized_sent = tokenizer.tokenize(sent)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
        segments_ids = [0] * len(tokenized_sent)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # for GPU computing, put everything on cuda
        if cuda:
            tokens_tensor = tokens_tensor.to('cuda')
            segments_tensors = segments_tensors.to('cuda')

        # predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)

        if type == 'vector':
            embedding.append(np.mean(encoded_layers[-1][0].cpu().numpy(), axis=0))
        else:
            embedding.append(encoded_layers[-1][0].cpu().numpy())

    return embedding
