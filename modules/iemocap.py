import os
import sys

import tensorflow as tf
import numpy as np 
import pandas as pd

from random import shuffle
from six.moves import xrange 
from itertools import islice



def read_labels(session_list, iemocap_dir, include_self_evaluation=True, apply_mapping=True):
    '''
    Read labels into a dataframe

    Returns: 
    A dataframe with 2 indices ('session' and 'turn') containing:
        - start_time
        - end_time
        - emotion
        - valence
        - activation
        - dominance 
    '''

    print('Reading labels...', end='', flush=True)
    data = {'start_time': [], 'end_time': [], 'name': [], 'emotion':[], 'valence': [], 'activation': [], 'dominance': []}

    if include_self_evaluation:
        data['val_scores'] = []
        data['act_scores'] = []
        data['dom_scores'] = [] 

    # mapping = {'neu' : 'Neutral', 'ang' : 'Angry', 'hap' : 'Happy', 'fea': 'Scared', 'sad' : 'Sad', 'exc': 'Excited', 'fru' : 'Frustrated',  
    #          'sur': 'Surprised', 'dis': 'Disgusted', 'oth': 'Other',  'xxx' : 'xxx'}

    mapping = {'neu' : 'LA', 'ang' : 'HN', 'hap' : 'HP', 'fea': 'xxx', 'sad' : 'LA', 'exc': 'HP', 'fru' : 'HN',  
               'sur': 'HP', 'dis': 'xxx', 'oth': 'xxx',  'xxx' : 'xxx'}

    for session_number in session_list:
        folder = iemocap_dir + 'Session' + str(session_number) + '\\dialog\\EmoEvaluation\\'
        for file in os.listdir(folder):
            if file.endswith('txt'):
                with open(folder + file) as f:
                    f.readline()
                    f.readline()

                    # parse utterances until end of file 
                    while True:
                        utter = []
                        line = f.readline()
                        
                        # Stop if the next line is empty 
                        if len(line) < 10:
                            break
                        else:
                            # Collect a block of lines describing one utterance 
                            while not line == '\n':
                                utter.append(line)
                                line = f.readline()

                            # save the parameters of the given utterance
                            words = utter[0].split('\t')
                            data['start_time'].append(np.float32(words[0].split()[0][1:]))
                            data['end_time'].append(np.float32(words[0].split()[2][:-1]))
                            data['name'].append(words[1])
                            data['emotion'].append(words[2])
                            data['valence'].append(np.float32(words[3][1:7]))
                            data['activation'].append(np.float32(words[3][9:15]))
                            data['dominance'].append(np.float32(words[3][17:23]))

                            val = []
                            act = []
                            dom = []

                            for e in range(1, len(utter)):
                                split = utter[e].split()
                                
                                if utter[e].startswith('A') and include_self_evaluation:
                                    val.append(np.float32(split[2][:-1]))
                                    act.append(np.float32(split[4][:-1]))
                                    if split[6] == ';':
                                        dom.append(np.NaN)
                                    else:
                                        dom.append(np.float32(split[6][:-1]))
                                elif utter[e].startswith('A') and utter[e][2] != 'F' and utter[e][2] != 'M':
                                    val.append(np.float32(split[2][:-1]))
                                    act.append(np.float32(split[4][:-1]))
                                    if split[6] == ';':
                                        dom.append(np.NaN)
                                    else:
                                        dom.append(np.float32(split[6][:-1]))
                            
                            if include_self_evaluation:
                                data['val_scores'].append(np.asarray(val))
                                data['act_scores'].append(np.asarray(act))
                                data['dom_scores'].append(np.array(dom))
    
    df = pd.DataFrame(data, dtype=np.float32)

    df['emotion'] = df['emotion'].replace(mapping)
    df['speaker'] = df['name'].copy().str[:-5] + '_' + df['name'].copy().str[-4]
    df['turn'] = df['name'].copy().str[-3:]
    # df['session'] = df['name'].copy().str[0:14]
    # df['turn'] = df['name'].copy().str[15:]
    df.set_index(['speaker', 'turn'], inplace=True)
    df.drop(['name'], axis=1, inplace=True)

    if include_self_evaluation:
        df['val_mean'] = df.val_scores.apply(np.mean)
        df['act_mean'] = df.act_scores.apply(np.mean)
        df['dom_mean'] = df.dom_scores.apply(np.mean)

        df['valence'] = df['val_mean']
        df['activation'] = df['act_mean']
        df['dominance'] = df['dom_mean']
        df.drop(['val_scores', 'act_scores', 'dom_scores', 'val_mean', 'act_mean', 'dom_mean'], axis=1, inplace=True)

    cols = ['start_time', 'end_time', 'emotion',
            'valence', 'activation', 'dominance']
    
    df = df[cols]
    print('             [Done]')
    return df

def read_lld_data(session_list, data_dir):
    c = 0
    for session_number in session_list:
        if c == 0:
            print('Reading data from sess {}...'.format(session_number), end='', flush=True),
            data = pd.read_csv(data_dir + 'Ses0' + str(session_number) + '_lld_IS11', sep=';')
            print('   [Done]   ')
            c = 1
        else:
            print('Reading data from sess {}...'.format(session_number), end='', flush=True),
            df= pd.read_csv(data_dir + 'Ses0' + str(session_number) + '_lld_IS11', sep=';')
            data = pd.concat([data, df])
            print('   [Done]   ')

    data['session'] = data['name'].copy().str[1:-6]
    data['turn'] = data['name'].copy().str[-5:-1]
    data.set_index(['session', 'turn', 'frameTime'], inplace=True)
    data.drop(['name'], axis=1, inplace=True)
    return data

def read_data(session_list, data_dir):
    c = 0
    for session_number in session_list:
        if c == 0:
            print('Reading data from sess {}...'.format(session_number), end='', flush=True),
            data = pd.read_csv(data_dir + 'Ses0' + str(session_number) + '_IS11', sep=';')
            print('   [Done]   ')
            c = 1
        else:
            print('Reading data from sess {}...'.format(session_number), end='', flush=True),
            df= pd.read_csv(data_dir + 'Ses0' + str(session_number) + '_IS11', sep=';')
            data = pd.concat([data, df])
            print('   [Done]   ')

    data['speaker'] = data['name'].copy().str[1:-6] + '_' + data['name'].copy().str[-5]
    data['turn'] = data['name'].copy().str[-4:-1]
    
    data.set_index(['speaker', 'turn'], inplace=True)
    
    return data.drop(['frameTime', 'name'],  axis=1)



