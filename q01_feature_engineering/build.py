# %load q01_feature_engineering/build.py
import pandas as pd
import numpy as np

battles = pd.read_csv('data/battles.csv')
character_predictions = pd.read_csv('data/character-predictions.csv')

def q01_feature_engineering(battles,character_predictions):
    'write your solution here'
    battles.loc[:, 'defender_count'] = (4 - battles[['defender_1', 'defender_2', 'defender_3', 'defender_4']].isnull().sum(axis = 1))
    battles.loc[:, 'attacker_count'] = (4 - battles[['attacker_1', 'attacker_2', 'attacker_3', 'attacker_4']].isnull().sum(axis = 1))
    battles.loc[:, 'att_comm_count'] = [len(x) if type(x) == list else np.nan for x in battles.attacker_commander.str.split(',')]
    character_predictions.loc[:, 'no_of_books'] = character_predictions[[x for x in character_predictions.columns if x.startswith('book')]].sum(axis = 1)  
    return(battles, character_predictions)

# q01_feature_engineering(battles,character_predictions)


