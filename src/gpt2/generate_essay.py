from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import tensorflow as tf
import numpy as np
from sentence_splitter import SentenceSplitter


# splitter = SentenceSplitter(language='ro')

# sentences = splitter.split(text)

def init_all(path_model, topics):
    tokenizer = GPT2Tokenizer.from_pretrained(path_model)
    topics = torch.tensor(tokenizer.encode(topics, add_special_tokens=True)).unsqueeze(0)
    model = GPT2LMHeadModel.from_pretrained(path_model)

    return model, topics, tokenizer


def token_to_sting(output, tokenizer):
    return tokenizer.decode(output, skip_special_tokens=True)


def generate_greedy(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, topics, min_len, max_len_generate,
                    repetition_penalty=1):
    topics = torch.tensor(tokenizer.encode(topics, add_special_tokens=True)).unsqueeze(0)
    model.eval()

    output = model.generate(topics, max_length=max_len_generate, min_length=min_len,
                            repetition_penalty=repetition_penalty)

    return token_to_sting(output[0], tokenizer)


def generate_beam_search(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, topics, min_len, max_len_generate, beam_width, no_repeat_ngram_size=2,
                         repetition_penalty=1):
    topics = torch.tensor(tokenizer.encode(topics, add_special_tokens=True)).unsqueeze(0)
    model.eval()

    output = model.generate(
        topics,
        max_length=max_len_generate,
        min_length=min_len,
        num_beams=beam_width,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=True,
        repetition_penalty=repetition_penalty
    )

    return token_to_sting(output[0], tokenizer)


def generate_sampling_top_k(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, topics, min_len, max_len_generate, no_repeat_ngram_size=2, top_k=10,
                            temperature=0.7, repetition_penalty=1):
    topics = torch.tensor(tokenizer.encode(topics, add_special_tokens=True)).unsqueeze(0)
    model.eval()
    # tf.random.set_seed(0)

    output = model.generate(
        topics,
        max_length=max_len_generate,
        min_length=min_len,
        top_k=top_k,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=True,
        repetition_penalty=repetition_penalty
    )

    return token_to_sting(output[0], tokenizer)


def get_len_token(string):
    return len(string.split(" "))
