import pandas as pd
import numpy as np

DATA_DIR = 'C:\\Users\\SMIL\\dev\\tf\\workspace\\features\\'

def emodb():
	"""
	Read in raw features, add metadata, normalize data

	Return (dataframe, list of features, list of speakers)

	"""
	mapping = {'F' : 0, 'N' : 1, 'W' : 2, 'T' : 3, 'A' : 4, 'L' : 5, 'E' : 6} # emotion labels
	data = pd.read_csv(DATA_DIR + 'emodb_lld', sep=';')
	data['speaker'] = data['name'].copy().str[1:3]
	data['utterance'] = data['name'].copy().str[3:6]
	data['label'] = data['name'].copy().str[6].replace(mapping)
	data['frameTime'] = data['frameTime'] * 100
	data['frameTime'] = data['frameTime'].astype(int)
	data.drop(['name'], axis=1, inplace=True)
	data.set_index(['speaker', 'utterance', 'frameTime'], inplace=True)

	features = data.drop(['label'], axis=1).columns
	speakers = list(data.index.get_level_values(0).unique())

	data[features] = (data[features] - data[features].mean()) / data[features].std()

	return data, features, speakers

def buemo():
	mapping = {'kiz' : 0, 'sev' : 1, 'ola' : 2, 'uzu' : 3} # emotion labels
	data = pd.read_csv(DATA_DIR + 'buemo_lld', sep=';')

	data['speaker'] = data['name'].copy().str[1:5]
	data['utterance'] = data['name'].copy().str[7:10]
	data['label'] = data['name'].copy().str[11:14].replace(mapping)
	data['frameTime'] = data['frameTime'] * 100
	data['frameTime'] = data['frameTime'].astype(int)
	data.drop(['name'], axis=1, inplace=True)
	data.set_index(['speaker', 'utterance', 'frameTime'], inplace=True)

	features = data.drop(['label'], axis=1).columns
	speakers = list(data.index.get_level_values(0).unique())

	data[features] = (data[features] - data[features].mean()) / data[features].std()

	return data, features, speakers

def read_ruslana():
	mapping = {'A' : 0, 'D' : 1, 'F' : 2, 'H' : 3, 'N' : 4, 'S' : 5} # emotion labels
	data = pd.read_csv(DATA_DIR + 'ruslana_lld_fixed.txt', sep=';')

	data['speaker'] = np.array(data['name'].copy().str[1:5])
	data['utterance'] = np.array(data['name'].copy().str[6:8])
	data['label'] = data['name'].copy().str[5].replace(mapping)
	data['frameTime'] = data['frameTime'] * 100
	data['frameTime'] = data['frameTime'].astype(int)
	data.drop(['name'], axis=1, inplace=True)
	data.set_index(['speaker', 'utterance', 'frameTime'], inplace=True)

	features = data.drop(['label'], axis=1).columns
	speakers = list(data.index.get_level_values(0).unique())

	data[features] = (data[features] - data[features].mean()) / data[features].std()

	return data, features, speakers