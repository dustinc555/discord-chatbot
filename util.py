# util.py
# set of utility functions to make things smoother
# includes:
# loadSettings, saveTokenizer, loadTokenizer

from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import os
import glob
import toml
import pickle
import math
import random

def loadSettings():
    '''returns settings dictionary must be in same dir as settings.toml to work'''
    f = open("settings.toml")
    settings_content = f.read()
    f.close()
    return toml.loads(settings_content)

settings = loadSettings()

def validate_texts(texts):
	'''texts: list of sentences
	returns a list of valid sentences'''
	return

def load_data_dictionary():
    '''returns a dictionary that contains:
		texts -> data as readable text
		tokens -> data processed as tokens'''
    tokenizer = loadTokenizer(settings['tokenizer']['production'])
    data = {}        
    raw_data = open(settings['data']['production']).readlines()
    max_word_count = settings['tokenizer']['max_word_count']
    oov_token = tokenizer.oov_token

    # convert raw text to arrays of words (sequences)
    raw_sequences = [text_to_word_sequence( line ) for line in raw_data]

    # remove questions and answers that are above the max size
    valid_data = []
    for i in range(0, len(raw_sequences), 2):
        if (len(raw_sequences[i]) <= max_word_count and len(raw_sequences[i + 1]) <= max_word_count):
                valid_data.append(raw_sequences[i])
                valid_data.append(raw_sequences[i + 1])

    # creates a map for words to tokens
    # higher frequency words will have a smaller token value
    # note: ensure char_level is set to false
    tokenizer.fit_on_texts(valid_data)

    # convert raw text to arrays of words (sequences)
    tokens = tokenizer.texts_to_sequences(valid_data)

    # pad to max size
    tokens = [sub_tokens + [tokenizer.oov_token] * (max_word_count - len(sub_tokens)) for sub_tokens in tokens]

    data['texts'] = [' '.join(sentence) for sentence in valid_data]
    data['tokens'] = np.array(tokens)
    return data


def load_and_process_data(path_to_data, tokenizer):
    '''returns (vocab, current, previous, anser_categorical)
    current and previous are vectors of word tokens,
    tokenizer: https://keras.io/preprocessing/text/
    unknown words are converted with tokenizer.oov_token'''

    raw_data = open(path_to_data).readlines()
    max_word_count = settings['tokenizer']['max_word_count']
    oov_token = tokenizer.oov_token

    # convert raw text to arrays of words (sequences)
    raw_sequences = [text_to_word_sequence( line ) for line in raw_data]

    # remove questions and answers that are above the max size
    valid_data = []
    for i in range(0, len(raw_sequences), 2):
        if (len(raw_sequences[i]) <= max_word_count and len(raw_sequences[i + 1]) <= max_word_count):
                valid_data.append(raw_sequences[i])
                valid_data.append(raw_sequences[i + 1])

    # creates a map for words to tokens
    # higher frequency words will have a smaller token value
    # note: ensure char_level is set to false
    tokenizer.fit_on_texts(valid_data)

    print(tokenizer.word_docs)

    # convert raw text to arrays of words (sequences)
    tokens = tokenizer.texts_to_sequences(valid_data)
    vocab = len(tokenizer.word_counts) + 2

    # pad to max size
    tokens = [sub_tokens + [tokenizer.oov_token] * (max_word_count - len(sub_tokens)) for sub_tokens in tokens]

    input_data = tokens[::2]
    next_data = tokens[1::2]

    context_user = [[tokenizer.oov_token] * max_word_count]
    for i in range(len(input_data) - 1):
        context_user.append(input_data[i])

    context_bot = [[tokenizer.oov_token] * max_word_count]
    for i in range(len(next_data) - 1):
        context_bot.append(next_data[i])

    # one hot encoding for answer set
    next_categorical = np.array([to_categorical(ans, num_classes=vocab) for ans in next_data])

    return (vocab, np.array(input_data), np.array(context_user), np.array(context_bot), next_categorical, next_data)

def load_training_data():
        '''returns the pickled data'''
        file_pi = open(settings['files']['data'], 'rb')
        return pickle.loads(file_pi.read())

def save_training_data(data):
        '''pickles the processed tokens'''
        file_handle = open(settings['files']['data'], 'wb')
        pickle.dump(data, file_handle)

def loadTokenizer(path):
    pickle_in = open(path, "rb")
    tokenizer = pickle.load(pickle_in)
    pickle_in.close()
    return tokenizer

def saveTokenizer(path, tokenizer):
    pickel_out = open(path, "wb")
    pickle.dump(tokenizer, pickel_out, protocol=pickle.HIGHEST_PROTOCOL)
    pickel_out.close()

def tokens_to_sentence(tokens, tokenizer):
    # remove oov_token
    tokens = list(filter(lambda x: x != tokenizer.oov_token, tokens))

    # 0 is invalid and will raise an error upon translation
    tokens = list(filter(lambda x: x != 0, tokens))

    # finally get sequences back from predicted tokens
    predicted_text = tokenizer.sequences_to_texts([tokens])

    # space out words
    return ' '.join(predicted_text)

def sentence_to_tokens(sentence):
	tokenizer = loadTokenizer(settings['tokenizer']['production'])
	max_word_count = settings['tokenizer']['max_word_count']

	word_sequence = text_to_word_sequence(sentence)

	tokens = tokenizer.texts_to_sequences([word_sequence])[0]

	return np.array(tokens + [tokenizer.oov_token] * (max_word_count - len(tokens)))

def predict_production(input_text="", context_user="", context_bot=""):
    ''' returns a reply to the given texts
 predicts according to settings
 model: production = 'models/production.h5'
 tokenizer: production = 'tokenizer/p_tokenizer_pi.obj'''

    model            = load_model(settings['model']['production'])
    tokenizer        = loadTokenizer(settings['tokenizer']['production'])
    max_word_count   = settings['tokenizer']['max_word_count']
    vocab            = len(tokenizer.word_counts) + 2

    # convert to sequences of text i.e. ["the", "dog", "barked"]
    word_sequences = [text_to_word_sequence(input_text), text_to_word_sequence(context_user), text_to_word_sequence(context_bot)]
    # convert to tokens: [4, 42, 85]
    tokens = tokenizer.texts_to_sequences(word_sequences)
    # pad tokens: [4, 42, 85, 1, 1, 1, 1, 1]
    padded_tokens = np.array([sub_tokens + [tokenizer.oov_token] * (max_word_count - len(sub_tokens)) for sub_tokens in tokens])


    input_text_idx = 0
    context_user_idx = 1
    context_bot_idx = 2
    prediction = model.predict([[padded_tokens[input_text_idx]],
                                [padded_tokens[context_user_idx]],
                                [padded_tokens[context_bot_idx]]])[0]

    # get most likley chars in the sequence from prediction
    predicted_tokens = [np.argmax(tok) for tok in prediction]

    print("predicted_tokens: " + str(predicted_tokens))

    # remove oov_token
    predicted_tokens = list(filter(lambda x: x != tokenizer.oov_token, predicted_tokens))

    # 0 is invalid and will raise an error upon translation
    predicted_tokens = list(filter(lambda x: x != 0, predicted_tokens))

    # finally get sequences back from predicted tokens
    predicted_text = tokenizer.sequences_to_texts([predicted_tokens])

    # space out words
    return ' '.join(predicted_text)

class KerasBatchGenerator(object):

    '''Used to fit the model
    this class shuffles the context and the response data to
    help the bot with transitions'''
    def __init__(self, context_questions, context_answers, input_data, answer_categorical, tokenizer, batch_size, vocab):
        self.context_questions = context_questions
        self.context_answers = context_answers
        self.input_data = input_data
        self.answers_categorical = answer_categorical
        self.batch_size = batch_size
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.current_idx = 0
        self.cq = np.full(self.context_questions.shape, self.tokenizer.oov_token)
        self.ca = np.full((self.batch_size,) + self.context_answers.shape, self.tokenizer.oov_token)
        self.ui = np.full((self.batch_size,) + self.context_answers.shape, self.tokenizer.oov_token)
        self.ac = np.zeros((self.batch_size,) + self.answers_categorical.shape)

    def generate(self):
        while True:
            # ca - context_answer
            # cq - context_question
            # -------------------------
            # ui - user_input
            # ac - answer_categorical
            # -------------------------
            # shuffle the context with the answers

            for i in range(self.batch_size):
                ca_cq_index = random.randint(0, len(self.context_answers) - 1)
                ui_ac_index = random.randint(0, len(self.input_data) - 1)

                while ui_ac_index != ca_cq_index:
                    ui_ac_index += 1
                self.cq[i] = self.context_questions[ca_cq_index]
                self.ca[i] = self.context_answers[ca_cq_index]
                self.ui[i] = self.input_data[ui_ac_index]
                self.ac[i] = self.answers_categorical[ui_ac_index]

            yield [self.cq, self.ca, self.ui], self.ac
