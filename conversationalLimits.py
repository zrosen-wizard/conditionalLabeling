import pandas as pd
import numpy as np
from toolkits.conditionalLabeling.booleanAlgebraMethod import *
from sklearn.metrics import cohen_kappa_score as kappa

def switch_none_type(i,array):
    if len(array) < 1:
        return [i]
    else:
        return array

df = pd.read_csv('/Users/zacharyrosen/Desktop/Example Corpus Method 1 - bq-results-20220923-132548-1663939575000.csv')
df = df.loc[:219].copy()
df['createdAt'] = pd.to_datetime(df['createdAt'])
z,r = df['Zachary Rosen'].astype(str).values, df['Richard Archer'].astype(str).values
# dfi = label_conversation_start_(df)
df['idx'] = df.index.values

conds = [
    lambda x: not bool(re.findall('(on its way|delivered|get back to you shortly)', x['body'].lower())),
    lambda x: (x['messageType'] in ['campaign', 'welcome', 'opt-in']) and (x['direction'] != 'inbound'),
    lambda x: (x['idx'] == df.loc[df['userNumber'].isin([x['userNumber']]) & df['serviceNumber'].isin([x['serviceNumber']])].index.min()),
    lambda x: (
                      x['createdAt'].tz_localize(None) - switch_none_type(df['createdAt'].loc[x['idx']].tz_localize(None), df['createdAt'].loc[
                                          df['userNumber'].isin([x['userNumber']])
                                          & df['serviceNumber'].isin([x['serviceNumber']])
                                          & df['direction'].isin(['outbound'])
                                          & (df['idx'] < x['idx'])
                  ].dt.tz_localize(None).values)[-1]
              ) > pd.Timedelta('1d')
]
df = label(df,conditions=conds,min_conditions_met=2)
df['11'] = df['11'].replace({True: 'start', False: np.nan})

bot = df['11'].astype(str).values
kappa(r,bot)