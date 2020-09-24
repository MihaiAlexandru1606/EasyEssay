import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import unicodedata
import re
import numpy as np
import os
import time
import random
from math import log
import json

from src.eval.usage import eval_essay

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()

    return w


def create_dataset(path, num_examples):
    phrases = []
    next_phrases = []
    topics = []

    max_len = -1
    with open(path, "r") as read_file:
        for line in read_file.readlines():
            line = line.strip()
            essay_body, topic = line.split("<topic>")
            current_phrase, next_phrase = essay_body.split("<next>")

            next_phrase = "<begin> " + next_phrase
            phrases.append(current_phrase)
            next_phrases.append(next_phrase)
            topics.append(topic)

            topics_token = topic.split()
            max_len = max(max_len, len(topics_token))

    return phrases, next_phrases, topics


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    start_lang, next_lang, topics = create_dataset(path, num_examples)
    start_tensor, start_tokenizer = tokenize(start_lang)
    next_tensor, next_tokenizer = tokenize(next_lang)
    topics_tensor, topics_tokenizer = tokenize(topics)

    return start_tensor, start_tokenizer, next_tensor, next_tokenizer, topics_tensor, topics_tokenizer


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, dropout=0.2):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       dropout=dropout
                                       )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)

        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class AttentionMTA(tf.keras.layers.Layer):

    def __init__(self, units):
        super(AttentionMTA, self).__init__()
        self.Ua = tf.keras.layers.Dense(units)
        self.Wa = tf.keras.layers.Dense(units)
        self.Va = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        hidden_state, topics_embedding = inputs
        hidden_state = tf.expand_dims(hidden_state, axis=1)

        # print(coverage_vector.shape)  # (batch_size, number_topics)
        # print(hidden_state.shape) # (batch_size, 1, units)
        # print(topics_embedding.shape) # (batch_size, number_topics, embedding_size)

        v = self.Va(tf.nn.tanh(self.Wa(hidden_state) + self.Ua(topics_embedding)))
        attention = tf.nn.softmax(v, axis=1)

        context = attention * topics_embedding
        context = tf.reduce_sum(context, axis=1)

        return context


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, dropout=0.2):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       dropout=dropout)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)
        self.attention_topics = AttentionMTA(self.dec_units)

    def call(self, x, hidden, enc_output, topics):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # topics shape == (batch_size, number_topics)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        topics_embedding = self.embedding(topics)
        contex_topics = self.attention_topics((hidden, topics_embedding))
        contex_topics = tf.expand_dims(contex_topics, axis=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), contex_topics, x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


class Seq2SeqMode(object):

    def __init__(self, path_config_file="../../config/seq2seq/config_seq2seq.json"):
        with open(path_config_file, 'r') as read_data:
            config = json.load(read_data)

        self.units = config["number_hidden_units"]
        self.embedding_size = config["embedding_dimension"]
        self.batch_size = config["batch_size"]
        self.max_generate = config["max_generation"]
        self.epochs = config["epochs"]
        self.start_token = config["start_seq"]
        self.end_token = config["end_seq"]
        self.step_to_save = config["steps_to_save"]
        self.path_save_model = config["path_save_model"]

        num_examples = 40000
        start_tensor, start_tokenizer, next_tensor, next_tokenizer, topics_tensor, topics_tokenizer = load_dataset(
            config["path_dataset"], num_examples)

        # TODO de modificat de aici
        self.start_tensor = start_tensor
        self.next_tensor = next_tensor
        self.topics_tensor = topics_tensor

        self.start_tokenizer = start_tokenizer
        self.next_tokenizer = next_tokenizer
        self.topics_tokenizer = topics_tokenizer

        # Calculate max_length of the target tensors
        self.len_topics = self.topics_tensor.shape[1]
        self.max_len_start = self.start_tensor.shape[1]
        self.max_len_next = self.next_tensor.shape[1]

        self.buffer_size = len(self.start_tensor)
        self.steps_per_epoch = len(self.start_tensor) // self.batch_size
        self.vocab_inp_size = len(self.start_tokenizer.word_index) + 1
        self.vocab_tar_size = len(self.next_tokenizer.word_index) + 1

        dataset = tf.data.Dataset.from_tensor_slices((self.start_tensor, self.next_tensor, self.topics_tensor)).shuffle(
            self.buffer_size)
        self.dataset = dataset.batch(self.batch_size, drop_remainder=True)

        self.encoder = Encoder(self.vocab_inp_size, self.embedding_size, self.units, self.batch_size, config["dropout"])
        self.decoder = Decoder(self.vocab_tar_size, self.embedding_size, self.units, self.batch_size, config["dropout"])

        # optimizator
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.checkpoint_dir = config["check_point_save"]
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    # def train_step(self, inp, targ, enc_hidden):
    def train_step(self, start_p, next_p, topics, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(start_p, enc_hidden)
            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([self.next_tokenizer.word_index['<begin>']] * self.batch_size, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, next_p.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output, topics)

                loss += self.loss_function(next_p[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(next_p[:, t], 1)

        batch_loss = (loss / int(next_p.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def train(self):

        for epoch in range(self.epochs):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (start_p, next_p, topics)) in enumerate(self.dataset.take(self.steps_per_epoch)):
                batch_loss = self.train_step(start_p, next_p, topics, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            if (epoch + 1) % self.step_to_save == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / self.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def _get_init(self, topics):
        topics = preprocess_sentence(topics)
        topics = [self.topics_tokenizer.word_index[i] for i in topics.split(' ')]
        topics = tf.keras.preprocessing.sequence.pad_sequences([topics], maxlen=self.len_topics, padding='post')
        topics = tf.convert_to_tensor(topics)

        inputs = tf.convert_to_tensor([[self.start_tokenizer.word_index['<start>'] for _ in range(self.max_len_start)]])
        hidden = tf.zeros((1, self.units))
        result = ''

        return topics, hidden, inputs, result

    def predict_greedy(self, topics):
        topics, hidden, inputs, result = self._get_init(topics)

        for _ in range(self.max_generate // self.max_len_next):
            enc_out, enc_hidden = self.encoder(inputs, hidden)

            dec_hidden = enc_hidden
            dec_inputs = tf.expand_dims([self.next_tokenizer.word_index['<begin>']], 0)
            next_input_encoder = []

            for t in range(self.max_len_next - 1):
                predictions, dec_hidden, _ = self.decoder(dec_inputs, dec_hidden, enc_out, topics)
                predict_id = tf.argmax(predictions[0]).numpy()

                result += self.next_tokenizer.index_word[predict_id] + ' '

                if self.next_tokenizer.index_word[predict_id] == '<stop>':
                    return result
                dec_inputs = tf.expand_dims([predict_id], 0)

                next_input_encoder += [predict_id]

            hidden = enc_hidden
            inputs = tf.convert_to_tensor([next_input_encoder])

        return result

    def _helper_beam_sear(self, input_seq, hidden_state, topics, beam_width):
        # # log_prob, seq index, seq string, coverage_vector, hidden_state
        dec_inputs = tf.expand_dims([self.next_tokenizer.word_index['<begin>']], 0)
        enc_out, enc_hidden = self.encoder(input_seq, hidden_state)
        beam_seqs = [(0.0, dec_inputs, hidden_state, [], [])]

        for _ in range(self.max_len_next):
            new_beam_seq = []
            for log_prob, dec_inputs, hidden_state, seq_string, seq_index in beam_seqs:
                predictions, dec_hidden, _ = self.decoder(dec_inputs, hidden_state, enc_out, topics)

                top_values, top_indices = tf.math.top_k(predictions[0], k=beam_width)
                top_values = top_values.numpy()
                top_indices = top_indices.numpy()

                for i in range(beam_width):
                    new_dec_input = tf.expand_dims([top_indices[i]], 0)
                    new_seq = seq_string + [self.next_tokenizer.index_word[top_indices[i]]]
                    seq_index = seq_index + [top_indices[i]]

                    if new_seq[-1] == "<end>":
                        return " ".join(new_seq), True, None

                    new_log_prob = log_prob - log(top_values[i])
                    new_beam_seq += [(new_log_prob, new_dec_input, dec_hidden, new_seq, seq_index)]

            ordered = sorted(new_beam_seq, key=lambda tup: tup[0])
            beam_seqs = ordered[:beam_width]

        return " ".join(beam_seqs[0][3]), False, tf.convert_to_tensor([beam_seqs[0][4]])

    def predict_beam_search(self, topics, beam_width):
        topics, hidden, inputs, result = self._get_init(topics)

        for _ in range(self.max_generate // self.max_len_next):
            seq_string, isFinal, seq_index = self._helper_beam_sear(inputs, hidden, topics, beam_width)
            result += seq_string

            if isFinal:
                return result

            inputs = seq_index

        return result

    def predict_sampling(self, topics, k, temperature):
        # tf.random.set_seed(1)
        topics, hidden, inputs, result = self._get_init(topics)
        # tf.random.set_seed(0)
        # np.random.seed(2**16-1)
        np.random.seed(0)
        for _ in range(self.max_generate // self.max_len_next):
            enc_out, enc_hidden = self.encoder(inputs, hidden)

            dec_hidden = enc_hidden
            dec_inputs = tf.expand_dims([self.next_tokenizer.word_index['<begin>']], 0)
            next_input_encoder = []

            for t in range(self.max_len_next - 1):
                predictions, dec_hidden, _ = self.decoder(dec_inputs, dec_hidden, enc_out, topics)

                top_values, top_indices = tf.math.top_k(predictions[0], k=k)
                top_values = tf.nn.softmax(top_values / temperature)
                top_values = top_values.numpy()
                top_indices = top_indices.numpy()

                predict_id = np.random.choice(top_indices, p=top_values)

                result += self.next_tokenizer.index_word[predict_id] + ' '

                if self.next_tokenizer.index_word[predict_id] == '<stop>':
                    return result, topics
                dec_inputs = tf.expand_dims([predict_id], 0)

                next_input_encoder += [predict_id]

            hidden = enc_hidden
            inputs = tf.convert_to_tensor([next_input_encoder])

        return result

    def restore(self):

        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def last_model(self):
        print(tf.train.latest_checkpoint(self.path_save_model))
        self.checkpoint.restore(tf.train.latest_checkpoint(self.path_save_model))

    def save(self):
        self.encoder.save_weights("./save_directory/encoder_{}.h5".format(time.time_ns()))
        self.decoder.save_weights("./save_directory/decoder_{}.h5".format(time.time_ns()))
