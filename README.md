# discord-chatbot
Keras project to build a model intended for use by a Discord bot.

# Environment
I have an amd gpu, so a docker container is provided that provides a suitable environment to utalize an amd gpu.
Depending on your hardware this may change.

# Process
The model learns on words, so I utalize Keras's tokenizer: https://keras.io/preprocessing/text/

An important thing to note though is that it cannot learn new words dynamically, words that it receives that are
unknown are given a special token. In order to learn more words the process needs to start over and the model
retrained. 


