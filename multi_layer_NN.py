from single_layer_NN import get_dep_indep_tensors, split_val_trn, reduce_indep_vars
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor 
from fastai.data.transforms import RandomSplitter
import torch.nn.functional as F
import pandas as pd


"""
Functions required to train a multi-layer neural network

Similar to a single layer NN except there are two layers and a constant.
The first layer is essentially a no_params X no_hidden matrix.
Second layer needs to then be multiplied by first layer and have no_params output (no_hidden X 1 matrix).
Constant term also required. 

Other functions altered to take into account multiple params.


"""

def init_params_ML(no_params_input, no_hidden_input = 20):
    l1 = (torch.rand(no_params_input, no_hidden_input)-0.5)/no_hidden_input #divide by number of hidden to keep similar magnitude
    l2 = torch.rand(no_hidden_input, 1)-0.3
    const = torch.rand(1)[0]
    return l1.requires_grad_(),l2.requires_grad_(),const.requires_grad_()

def get_preds_ML(l1_input, l2_input, const_input, td_i_input):
    preds = F.relu(td_i_input.double()@l1_input.double()) #relu activation function
    preds = preds@l2_input.double() + const_input
    return torch.sigmoid(preds)

def loss_func_ML(l1_input,l2_input,const_input, td_i_input, td_d_input): 
    return torch.abs(get_preds_ML(l1_input,l2_input,const_input, td_i_input)-td_d_input).mean()

def update_params_ML(l1_input, l2_input, const_input, lr_input):
    params = l1_input,l2_input,const_input
    for layer in params:
       layer.sub_(layer.grad * lr_input)
       layer.grad.zero_()
       
def run_ML(l1_input,l2_input,const_input, lr_input, td_i_input, td_d_input):
    loss = loss_func_ML(l1_input,l2_input, const_input, td_i_input, td_d_input)
    loss.backward()
    with torch.no_grad(): 
        update_params_ML(l1_input,l2_input,const_input, lr_input)
        print(f"{loss:.3f}", end = '; ')

def train_model_ML(td_i_input, td_d_input, runs_input = 1500, lr_input = 50, no_params_input = 10, batches = False, batch_size = 100):
    torch.manual_seed(500)
    l1, l2, const = init_params_ML(no_params_input)
    if batches == True:
        for i in range(runs_input): 
            batch_index = np.random.randint(len(td_i_input), size = batch_size)
            run_ML(l1,l2,const, lr_input, td_i_input[batch_index], td_d_input[batch_index])
    else:
        for i in range(runs_input): 
            run_ML(l1,l2,const, lr_input, td_i_input, td_d_input)
    loss = loss_func_ML(l1,l2,const, td_i_input, td_d_input)
    return l1,l2,const, loss

"""
Produces submission file, and gives accuracy if it is not a test file (and you know outcomes)
"""

def final_results_ML (l1_input, l2_input, const_input, d_i_input, d_d_input, passenger_id_input, submission_input = True, accuracy_input = True):
    transported_results = get_preds_ML (l1_input, l2_input, const_input, d_i_input) > 0.5

    if accuracy_input == True:
        accuracy = (d_d_input.bool()==transported_results).float().mean()

    if submission_input == True:
        t1 = transported_results.view(-1,1)
        flat_t1 = [item for sublist in t1 for item in sublist]
        flat_t1 = [True if tensor == torch.tensor(True) else False for tensor in flat_t1]
        data =  {'PassengerId': passenger_id_input, 'Transported': flat_t1}
        t3 = pd.DataFrame(data)
        t3.to_csv('submission.csv', index = False)
        
    if accuracy_input == True:
        return accuracy





