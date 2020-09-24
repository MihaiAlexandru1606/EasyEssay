import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import BertForMaskedLM, BertTokenizer
import tensorflow as tf
import numpy as np
import torch

ID_MASK = 5
ID_SEP = 4
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"


def get_id_mask(tokens):
    mask_ids = []
    for i, token in enumerate(tokens):
        if token == MASK_TOKEN:
            mask_ids.append(i)

    return mask_ids


def get_predict_greedy(prediction):
    return tf.argmax(prediction)


def get_prediction_sampling(prediction, k, temperature):
    top_values, top_indices = tf.math.top_k(prediction, k=k)
    top_values = tf.nn.softmax(top_values / temperature)
    top_values = top_values.numpy()
    top_indices = top_indices.numpy()

    predicted_id = np.random.choice(top_indices, p=top_values)

    return predicted_id


def merge_token(tokens):
    list_word = []

    for token in tokens:
        if token[0] == '#':
            list_word[-1] = list_word[-1] + token[2:]
        else:
            list_word.append(token)

    return " ".join(list_word)


def generate_essay(path_to_model, topics, begin_sequence, len_generate, strategy="greedy", k=5, temperature=0.7):
    bert = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=path_to_model, output_attentions=False)
    bert.eval()
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=path_to_model)

    number_generation_word = 0
    tokenized_text = tokenizer.tokenize("[CLS] " + topics + " [SEP] " + begin_sequence + " [MASK] " + " [SEP]")

    while number_generation_word < len_generate:
        mask_ids = get_id_mask(tokenized_text)

        temp_token_text = tokenized_text.copy()
        temp_token_index_text = tokenizer.convert_tokens_to_ids(temp_token_text)
        id_mask = [idx for idx, i in enumerate(temp_token_index_text) if i == ID_MASK]

        temp_segments_ids = [0] * len(temp_token_text)
        tokens_tensor = torch.tensor([temp_token_index_text])
        segments_tensors = torch.tensor([temp_segments_ids])

        with torch.no_grad():
            outputs = bert(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        predicted_index = ID_MASK
        predictions = predictions[0, id_mask][0]
        if strategy == "greedy":
            predicted_index = get_predict_greedy(predictions)
        elif strategy == "sampling":
            predicted_index = get_prediction_sampling(predictions, k, temperature)

        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        tokenized_text[mask_ids[0]] = predicted_token
        tokenized_text = tokenized_text[0:-1] + ['[MASK]'] + [tokenized_text[-1]]

        number_generation_word += 1

    return merge_token(tokenized_text)


if __name__ == '__main__':
    topics = "Particularități de construcție a personajului o scrisoare pierduta Ion Luca Caragiale comedie genul dramatic "
    start = "Particularități de construcție a personajului  Statutul "
    target = "social al lui Nae Cațavencu este precizat în lista personajelor"

    path_to_bert = "../../model/bert/bert-v7"
    print(generate_essay(path_to_bert, topics, start, 5, strategy="greedy"), " Greedy ")
    print(generate_essay(path_to_bert, topics, start, 5, strategy="sampling", k=7, temperature=0.7),
          " Sampling 7, 0,7 ")
    print(generate_essay(path_to_bert, topics, start, 5, strategy="sampling", k=5, temperature=0.5),
          " Sampling 5, 0,5 ")
    print(generate_essay(path_to_bert, topics, start, 5, strategy="sampling", k=10, temperature=0.7),
          " Sampling 10, 0,7 ")

    path_to_bert = "../../model/bert/bert-readerbench"
    print(generate_essay(path_to_bert, topics, start, 5, strategy="greedy"), " Greedy ")
    print(generate_essay(path_to_bert, topics, start, 5, strategy="sampling", k=7, temperature=0.7),
          " Sampling 7, 0,7 ")
    print(generate_essay(path_to_bert, topics, start, 5, strategy="sampling", k=5, temperature=0.5),
          " Sampling 5, 0,5 ")
    print(generate_essay(path_to_bert, topics, start, 5, strategy="sampling", k=10, temperature=0.7),
          " Sampling 10, 0,7 ")
