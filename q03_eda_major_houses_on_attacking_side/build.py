# %load q03_eda_major_houses_on_attacking_side/build.py
import pandas as pd
import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from greyatomlib.game_of_thrones.q01_feature_engineering.build import q01_feature_engineering
# plt.switch_backend('agg')

battles = pd.read_csv('data/battles.csv')
character_predictions = pd.read_csv('data/character-predictions.csv')

battle,cp = q01_feature_engineering(battles, character_predictions)


def q03_eda_major_houses_on_attacking_side(df):
    'write your solution here'
    a = df['attacker_count'].value_counts().sort_index().plot(kind='bar',rot=0)

# q03_eda_major_houses_on_attacking_side(battle)


