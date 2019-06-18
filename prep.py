#!/usr/bin/python
# prep.py
# reads all sets listed in settings.toml
# and converts them to a main data file
# the main data file has the format: (in sentence)\n(out sentence)\n
# this file is not a library file it must be run
import toml
import util
import glob
import yaml
import pandas as pd

settings = util.loadSettings()
datasets_dir = settings['directories']['datasets']


def massage_chatterbot():
	# load files
	file_paths = glob.glob(datasets_dir + "/chatterbot/*.yml")

	# 2d array of size 2 arrays [Q, A]
	data_arr = []

	# combine all conversations
	for fp in file_paths:
		df = pd.io.json.json_normalize(yaml.load( open(fp) ))
		data_arr.extend(df['conversations'][0])

	# clean up strings and combine
	return ''.join(str(sentence).replace('\n', ' ').strip() + '\n' for conversation in data_arr for sentence in conversation)

# event dispatch table for loading datasets
data_preperation_procedures = {
	"chatterbot" : massage_chatterbot
}



data_file_content = ""

# merge all datasets into single source
for data_set in settings['preperation']['sets']:
	data_file_content += data_preperation_procedures[data_set]()

#save parsed data
data_file = open(settings['files']['training'], "w")
data_file.writelines(data_file_content)
data_file.close()