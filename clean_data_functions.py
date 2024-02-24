"""

Contains functions to clean the training data.
Outputs clean and processed data for single-layer neural network. 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#If the passenger is in cryosleep or age is less than 13 then expentiture is zero 
def fill_NaNs_expenditure (df_input):
    expenditure_cats = ['RoomService', 'VRDeck', 'Spa', 'ShoppingMall', 'FoodCourt']
    for cat in expenditure_cats:
        df_input[cat] = np.where((df_input['Age']<13) | (df_input['CryoSleep']==True), 0, df_input[cat])
    return df_input

#Fill the NaNs with the most common (mode) data in that column
def fill_NaNs_mode (df_input):
    df_input.fillna(df_input.mode().iloc[0], inplace=True)
    return df_input

#Cabin is formatted as such LETTER/NUMBER/LETTER - so split into three different parameters
def split_cabin (df_input):
    df_input[['Cabin1', 'Cabin2', 'Cabin3']] = df_input['Cabin'].str.split('/', expand=True)
    df_input.drop(columns=['Cabin'], inplace = True)
    return df_input

#Create dummy variables for categorical parameters 
#We can also drop one column of each categorical variable without losing any information
def get_dummy_vars (df_input):
    df_input = pd.get_dummies(df_input, columns=['Cabin1', 'Cabin3', 'HomePlanet', 'Destination'])
    df_input.drop(columns = ['Cabin1_T', 'HomePlanet_Mars', 'Destination_PSO J318.5-22'], inplace = True)
    return df_input

#Drop 'Name' parameter for now
#Later, it's possible to extract extra information like family size 
def drop_name (df_input):
    df_input.drop(columns = ['Name'], inplace = True)
    return df_input

#To reduce the number of columns, we can add up all of the expenditures
#This reduces data, but we can reintroduce it later
#May be better to have lower number of parameters to reduce overfitting
#Take log of expenditures (small no of people can spend a lot more than others)
def sum_expenditures(df_input):
    expenditure_cats = ['RoomService', 'VRDeck', 'Spa', 'ShoppingMall', 'FoodCourt']
    df_input['Expenditures'] = 0
    for cat in expenditure_cats:
        df_input['Expenditures'] += df_input[cat]
        df_input.drop(columns = [cat], inplace = True)
    df_input['Expenditures'] = np.log(df_input['Expenditures']+1)
    return df_input

#Normalise numerical data so it's between 0 and 1
#So when training, the numerically larger data does not overwhelm smaller data

def normalise_data(df_input):
    df_input['Expenditures'] = df_input['Expenditures']/max(df_input['Expenditures'])
    df_input['Age'] = df_input['Age']/max(df_input['Age'])
    df_input['Cabin2'] = df_input['Cabin2'].astype(float)/max(df_input['Cabin2'].astype(float))
    return df_input

#Final function to clean data
def clean_data(df_input, to_csv = True, test = True):
    df_input = fill_NaNs_expenditure(df_input)
    df_input = fill_NaNs_mode(df_input)
    df_input = split_cabin (df_input)
    df_input = get_dummy_vars (df_input)
    df_input = drop_name (df_input)
    df_input = sum_expenditures(df_input)
    df_input = normalise_data(df_input)
    if to_csv == True and test == True:
        df_input.to_csv('clean_test_data.csv')
    if to_csv == True and test == False:
        df_input.to_csv('clean_training_data.csv')
    return df_input

