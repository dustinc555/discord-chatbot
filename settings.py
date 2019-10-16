
# raw unprepared data
raw_datasets_dir = "data/sets"

# path to prepared data
data_file_path = 'data/train/data.txt'

data = 'data/train/data_pi.obj'

# sets to use when preparing the data for the data file
sets = ["chatterbot", "custom"]


# tokenizer stuff
# unfortunatly, there needs to be some way to save the tokenizer
# that is bounded to a given model, they need each other, one manipulates tokens
# the other understands what they mean
tokenizer_obj_path = 'tokenizer/p_tokenizer_pi.obj'

# this is the special token given to unknown words
oov_token = 1

# number of tims the model feeds into itself for a response
# in the future I am not going to be doing it by sentences
# I am going to add a start of and end of sentence token and
# I am going to call upon the model until an end of sentence token
# is given.
max_word_count = 30

# required by the tokenizer, make sure this is greater than
# the actual number of words in the data
num_words = 6000

# path for active model
production_model_path = 'models/production.h5'


# training variables
batch_size = 100
epochs = 1000
