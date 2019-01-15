# %load q05_number_of_commanders/build.py
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('white')
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from greyatomlib.game_of_thrones.q01_feature_engineering.build import q01_feature_engineering
# plt.switch_backend('agg') 

battles = pd.read_csv('data/battles.csv')
character_predictions = pd.read_csv('data/character-predictions.csv')

battle, p = q01_feature_engineering(battles,character_predictions)
def q05_number_of_commanders(battles):
    'write your solution here'
    sns.boxplot(x='attacker_king',y='att_comm_count',data=battle)
    return

# q05_number_of_commanders(battle)


