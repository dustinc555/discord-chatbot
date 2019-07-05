# discord-chatbot
Keras project to build a model intended for use by a Discord bot.

# Environment
I have an amd gpu, so a docker container is provided that provides a suitable environment to utilize  an amd gpu.
Depending on your hardware this may change.

# Process
The words need to be converted to tokens in order to learn on them.
I utilize Keras's tokenizer: https://keras.io/preprocessing/text/.

Note: The model cannot learn on words dynamically, words that it receives that are
unknown are given a special token. In order to learn more words the process needs to start over and the model
retrained. 

The following process depends on the contents of settings.toml

1. Data is prepared by running python3 prep.py 
2. The model is created by running python3 makemodel.py, produces models/production.h5
3. The model may continue to be trained by running python3 train.py, trains models/production.h5

# Data
The data trained on is located in data/train/data.txt; it is specified in settings.toml under files->training.
The file consists of a list of sentences seperated by new lines. Every other sentence is a reply to the previous.
*The end of sentence delimiter is the newline character.*

Every dataset name listed in settings.toml->preperation->sets is used as a key in ```data_preperation_procedures``` to call a function that prepares the data.

## Adding sets
1) Place the data set to its own sub folder in data/sets.
2) Add a unique name for the set in settings.toml->preperation->sets.
3) Create a function that also has a unique name in preperation->sets in prep.py that parces the data.
4) Add the function name to ```data_preperation_procedures``` in prep.py as a value with the name in step 2 as the key.  

The function added in step 3 returns the data as (input sentence)\n(output sentence)\n ...
This will be summed up with the output of the other sets and consolidated in data/train/data.txt.

## Recommended sets
* chatterbot: https://www.kaggle.com/kausr25/chatterbotenglish#botprofile.yml

# The Model
## LSTM - Long short term memory


