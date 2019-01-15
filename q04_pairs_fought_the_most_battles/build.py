# %load q04_pairs_fought_the_most_battles/build.py
import pandas as pd
from collections import Counter
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from greyatomlib.game_of_thrones.q01_feature_engineering.build import q01_feature_engineering

import matplotlib.pyplot as plt
plt.switch_backend('agg') 

battles = pd.read_csv('data/battles.csv')
character_predictions = pd.read_csv('data/character-predictions.csv')

def q04_pairs_fought_the_most_battles(data):
    'write your solution here'
    # Ignoring records where either attacker_king or defender_king is null. Also ignoring one record where both have the same value.
    

    df = (list(Counter([tuple(set(x)) for x in battles.dropna(subset=['attacker_king', 'defender_king'])[['attacker_king', 'defender_king']].values if len(set(x)) > 1]).items()))
    df1 = pd.DataFrame(df)
    
    ax = df1.plot(kind='bar',title ='', figsize=(15, 10), legend=True)
    ax.set_xlabel('No of battles', fontsize=12)
    ax.set_ylabel('No of Death/Capture Event', fontsize=12)
    plt.show()
    return (df)
    
# q04_pairs_fought_the_most_battles(battles)


