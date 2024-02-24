import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor 
from fastai.data.transforms import RandomSplitter
import torch.nn.functional as F
import pandas as pd

"""
Import required functions from the different files
"""

from clean_data_functions import *
from single_layer_NN import *
from multi_layer_NN import *

#Import and clean the data for training and test sets

df = pd.read_csv('train.csv')
df = clean_data(df, to_csv = True, test = False)

df = pd.read_csv('clean_training_data.csv')

df_test = pd.read_csv('test.csv')
df_test = clean_data(df_test, to_csv = True, test = True)
df_test = pd.read_csv('clean_test_data.csv')

#Get indep and dep var for training and test

d_i, d_d = get_dep_indep_tensors(df, test = False)
passenger_ids = get_passenger_ids(df)

test_i= get_dep_indep_tensors(df_test, test = True)
test_passenger_ids = get_passenger_ids(df_test)


#Split data into validation and training sets

td_i, vd_i, td_d, vd_d, t_passenger_ids, v_passenger_ids = split_val_trn (d_i, d_d, df, passenger_ids)

#Choose the most important columns 

NO_RUNS = 100
LR = 50

loss_list, index_list = get_lowest_loss_list(td_i, td_d, NO_RUNS, LR, no_params_input=td_i.shape[1])
print('Columns that decrease loss most to least: ')
print(index_list) #Can refer back to csv files to see which columns these indices refer to

record_loss, record_columns_index = get_losses_for_columns(td_i, td_d, NO_RUNS, LR)
print('The loss is: ')
print(record_loss)
print('After adding on these respective columns and training.')
print('Columns: ', record_columns_index)

np.savetxt('important_columns_index.csv', record_columns_index)

#Reduce number of columns used to train (here, using 10 most important)

td_i = reduce_indep_vars(td_i, record_columns_index[:10])
vd_i = reduce_indep_vars(vd_i, record_columns_index[:10])

test_i = reduce_indep_vars(test_i, record_columns_index[:10])

#Reformat d_d

td_d = td_d[:,None]
vd_d = vd_d[:,None]

#Train model

NO_RUNS = 2000
LR = 50

l1,l2, const, loss_ML = train_model_ML(td_i, td_d, NO_RUNS, LR, batches = True)
final_results_ML(l1,l2,const, test_i, vd_d, test_passenger_ids, True, False)

print('Completed')

