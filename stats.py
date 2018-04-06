import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from modules import iemocap

label_dir = 'D:\\IEMOCAP_full_release\\'

train_sessions = [1, 2, 3, 4, 5]
data = iemocap.read_labels(train_sessions, label_dir, include_self_evaluation=False, apply_mapping=True)

valence = data['valence']
activation = data['activation']

# valence = [(lambda val: 0 if val <= 2.3 else (2 if val >= 3.3 else 1))(val) for val in valence]
# activation = [(lambda val: 0 if val <= 2 else (2 if val >= 3 else 1))(val) for val in activation]

# plt.hist(activation, 3)
# plt.show()

grp = data.groupby(level=[0])

turns = []
for dialog, df in grp:
	turns.append(len(df))

plt.hist(turns, 20)
plt.show()
