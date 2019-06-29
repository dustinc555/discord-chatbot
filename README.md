# discord-chatbot
Keras project to build a model intended for use by a Discord bot.

# Environment
I have an amd gpu, so a docker container is provided that provides a suitable environment to utilize  an amd gpu.
Depending on your hardware this may change.

# Process
The model learns on words, so I utilize Keras's tokenizer: https://keras.io/preprocessing/text/

An important thing to note though is that it cannot learn new words dynamically, words that it receives that are
unknown are given a special token. In order to learn more words the process needs to start over and the model
retrained. 

The following process depends on the contents of settings.toml

1. Data is prepared by running python3 prep.py 
2. The model is created by running python3 makemodel.py, produces models/production.h5
3. The model may continue to be trained by running python3 train.py, trains models/production.h5