import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor 
from fastai.data.transforms import RandomSplitter
import torch.nn.functional as F
import pandas as pd

"""
Splits the clean data into independant and dependent (Transported) data.
"""

def get_dep_indep_tensors (df_input, test = False):
    columns_i = list(df_input.columns)
    if test == False:
        columns_i.remove('Transported')
    columns_i.remove('PassengerId')
    columns_i.remove('Unnamed: 0')
    df_input = df_input.astype(float)

    if test == False:
        d_d = tensor(df_input['Transported'])
    d_i = tensor(df_input[columns_i].values)
    
    if test == False:
        return d_i, d_d
    
    if test == True:
        return d_i
    
"""
Returns list of passenger ids (required for submission)
"""
def get_passenger_ids (df_input):
    return df_input['PassengerId']


"""
Split the data into validation and training sets
"""

def split_val_trn (d_i_input, d_d_input, df_input, passenger_ids_input, seed_input = 42):
    t_split, v_split = RandomSplitter(seed = seed_input)(df_input) #Just creates a random split
    td_i = d_i_input[t_split]
    vd_i = d_i_input[v_split]
    td_d = d_d_input[t_split]
    vd_d = d_d_input[v_split]

    t_passenger_ids = passenger_ids_input[t_split]
    v_passenger_ids = passenger_ids_input[v_split]

    return td_i, vd_i, td_d, vd_d, t_passenger_ids, v_passenger_ids



"""
Functions required to train a single layer neural network

Independant data is a matrix with columns corresponding to different parameters (eg. Age, CryoSleep) and rows for each passenger
With single layer NN, we want to fit some parameters (vector of size = number of columns of data matrix)
Such that if we matrix multiply data matrix by fitted parameters, we get a good predictor of dependant data
Can use loss function and calculate gradients of the parameters
Then must minimise the loss function 

1. init_params: starting parameters
2. get_preds: calculate the predicted outcomes (transported or not) by matrix multiplying data and params 
3. loss_func: loss is abs(predicted - actual).mean()
4. update_params: subtract grad of params from params to get new params 

Now we want to continue doing this over multiple runs 

5. run: updates the params once. Option to run in batches to decrease overfitting
6. train_model: can do many runs and outputs loss and final predicteds

Some functions require specificaion of number of params used (not for multi-layer NN, though)
This is only because we use the single layer NN to find the most important data columns

"""

def init_params(no_params_input):
    return (torch.rand(no_params_input)-0.5).requires_grad_() #centralise params around zero

def get_preds(params_input, td_i_input, no_params_input):
    if no_params_input > 1: 
        return torch.sigmoid((td_i_input * params_input).sum(axis = 1)) #sigmoid to get between 0 and 1
    else:
        return torch.sigmoid(td_i_input * params_input) 
    
def loss_func(params_input, td_i_input, td_d_input, no_params_input):
    return torch.abs(get_preds(params_input, td_i_input, no_params_input)-td_d_input).mean()

def update_params(params_input, lr_input):
    params_input.sub_(params_input.grad * lr_input)
    params_input.grad.zero_() 

def run(params_input, lr_input, td_i_input, td_d_input, no_params_input):
    loss = loss_func(params_input, td_i_input, td_d_input, no_params_input)
    loss.backward()
    with torch.no_grad(): 
        update_params(params_input, lr_input)
        print(f"{loss:.3f}", end="; ")

def train_model(td_i_input, td_d_input, runs_input = 1500, lr_input = 50, no_params_input = 18, batches = False, batch_size = 100):
    torch.manual_seed(500)
    params = init_params(no_params_input)
    if batches == True:
        for i in range(runs_input): 
            batch_index = np.random.randint(len(td_i_input), size = batch_size)
            run(params, lr_input, td_i_input[batch_index], td_d_input[batch_index], no_params_input)
    else:
        for i in range(runs_input): 
            run(params, lr_input, td_i_input, td_d_input, no_params_input)
    loss = loss_func(params, td_i_input, td_d_input, no_params_input)
    return params, loss


"""

Choosing columns

It would be better to reduce the number of parameters used to train the model.
Some data columns may not affect the outcome at all, and these can be removed. 
Method:
- Train the model with each column seperately 
- Find the one that decreases the loss the most 
- Add that column index to a list
- Train the model again, with the previous column and each one of the others
- Keep going until the loss decreases only trivially
- Train with all data (not batches) to reduce randomness in choosing columns

Functions required:
1. get_lowest_loss_one_column: finds the data column that gives the lowest loss 
2. get_lowest_loss_multi_columns: can take multiple data columns and run with an additional one to find additional column that decreases the loss the most
3: get_losses_for_columns: orders the columns in terms of which additional column decreases the loss the most

Also to get losses for each column seperately (as if training model with that one column)
4. get_lowest_losses_list: outputs a losses and column index
"""

def get_lowest_loss_one_column (td_i_params_input, td_d_input, NO_RUNS_input, LR_input, no_params_list_input):
    loss_counter = float('inf')
    index_counter = 0
    for i in no_params_list_input:
        params, loss = train_model(td_i_params_input[:,i], td_d_input, NO_RUNS_input, LR_input, 1)
        if loss < loss_counter:
            loss_counter = loss
            index_counter = i
    
    return loss_counter, index_counter 

def get_lowest_loss_multi_columns (td_i_params_input, td_d_input, NO_RUNS_input, LR_input, no_params_list_input, merged_params_input, counter_input):
    loss_counter = float('inf')
    index_counter = 0
    for i in no_params_list_input:
        t1 = td_i_params_input[:,i].view(-1,1)
        t2 = merged_params_input
        t3 = (torch.cat((t1,t2), dim = 1))
        
        params, loss = train_model(t3, td_d_input, NO_RUNS_input, LR_input, (counter_input+1))
        if loss < loss_counter:
            loss_counter = loss
            index_counter = i
    
    return loss_counter, index_counter

def get_losses_for_columns(td_i_input, td_d_input, NO_RUNS_input, LR_input):
    no_params_list = list(range(td_i_input.shape[1]))
    record_loss = np.zeros(td_i_input.shape[1])
    record_params_index = np.zeros(td_i_input.shape[1])

    record_loss[0], record_params_index[0] = get_lowest_loss_one_column(td_i_input, td_d_input, NO_RUNS_input, LR_input, no_params_list)
    
    no_params_list.remove(record_params_index[0]) #remove most important param from list 
    merged_params = td_i_input[:, int(record_params_index[0])].view(-1,1) #make into column format

    counter = 0 #to put objects in correct index 
    for j in range(len(no_params_list)):
        #if i in no_params_list:
        counter += 1
        record_loss[counter], record_params_index[counter] = get_lowest_loss_multi_columns(td_i_input, td_d_input, NO_RUNS_input, LR_input, no_params_list, merged_params, counter)
        to_merge_param = td_i_input[:,int(record_params_index[counter])].view(-1,1)
        merged_params = (torch.cat((to_merge_param, merged_params), dim = 1))
        no_params_list.remove(record_params_index[counter])
    return record_loss, record_params_index

def get_lowest_loss_list (td_i_params_input, td_d_input, NO_RUNS_input, LR_input, no_params_input):
    loss_list = np.zeros(no_params_input)
    index_list = np.zeros(no_params_input)

    for i in range(no_params_input):
        params, loss = train_model(td_i_params_input[:,i], td_d_input, NO_RUNS_input, LR_input, 1)
        loss_list[i]=loss
        index_list[i]=i
    
    return loss_list, index_list

"""
Create a new tensor just using chosen/most important columns
"""
def reduce_indep_vars (d_i_input, column_index_input):
    new_d_i = d_i_input[:, column_index_input]
    return new_d_i




