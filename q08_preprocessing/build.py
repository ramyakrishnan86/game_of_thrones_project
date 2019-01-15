# %load q08_preprocessing/build.py
import pandas as pd
import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from greyatomlib.game_of_thrones.q01_feature_engineering.build import q01_feature_engineering
from greyatomlib.game_of_thrones.q07_culture_survival.build import q07_culture_survival

battles = pd.read_csv('data/battles.csv')
character_predictions = pd.read_csv('data/character-predictions.csv')

battle, character_pred = q01_feature_engineering(battles,character_predictions)
def q08_preprocessing(data):
    #'write your solution here'
    colu = ['title','culture','mother','father','house','heir' ,'spouse']
    for col in colu:
        data[col] = pd.factorize(data[col])[0]
    col_1 = ['name', 'alive', 'pred', 'plod', 'isAlive', 'dateOfBirth']
    data = data.drop(col_1, 1)   
    col = data.columns
    for c in col:
        data[c] = data[c].replace(['.', '_'], ' ')
    for c in col:
        data[c] = data[c].replace(np.nan, -1)
    
    return (data)

# q08_preprocessing(character_pred)


