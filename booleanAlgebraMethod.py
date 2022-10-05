import numpy as np
import pandas as pd
import datetime as dt
import regex as re


####### Example Usage #######
"""
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
df = label_conversation_start(df,conditions=conds)
"""
#############################

def complex_index(x, df, delim_cols=['userNumber', 'serviceNumber'], index_ref='idx', direction = lambda x,y: x < y, selection = lambda z: max(z)):
    """
    Note: the configuration
                direction = lambda x,y: x < y, selection = lambda z: max(z)
          yields the most recent, prior index. While
                direction = lambda x,y: x > y, selection = lambda z: min(z)
          yields the soonest, next index.


    :param x: the dataframe subset . . . think "x" from .apply(lambda x: 0)
    :param df: the total dataframe
    :param delim_cols: Uses these columns in x to find other rows in the dataframe
    :param index_ref: the index column. Necessary to find previous or proceeding instances.
    :param direction: a function that sets whether we're looking at indexes lower or higher than x's
    :param selection: a function that sets whether we find the max or min in the direction.
    :return:
    """
    i = x[index_ref]
    sel = np.prod([df[col].isin([x[col]]) for col in delim_cols],axis=0) == 1
    seli = direction(df.index.values, i)

    possibilities = df.loc[sel & seli].index.values
    if len(possibilities) > 0:
        return selection(possibilities)
    else:
        return i


def label(df,
          conditions,
          un_col = 'userNumber',
          sn_col = 'serviceNumber',
          resort_by_columns=['userNumber', 'serviceNumber', 'createdAt'],
          min_conditions_met=None
          ):
    """

    Example complex condition:
    lambda x: (x['direction']=='inbound') and (df['direction'].loc[complex_index(x,df)]=='outbound')

    :param df:
    :param conditions:
    :param un_col:
    :param sn_col:
    :param resort_by_columns:
    :param min_conditions_met:
    :return:
    """

    min_hits = min_conditions_met
    if min_conditions_met == None:
        min_hits=len(conditions)

    dfi = df.sort_values(by=resort_by_columns).copy()
    dfi.index=range(len(dfi))

    new_col = str(len(list(dfi)))
    dfi[new_col] = False

    for customer,agent in dfi[[un_col,sn_col]].drop_duplicates().values:
        subdata = dfi.loc[dfi[un_col].isin([customer]) & dfi[sn_col].isin([agent])]
        sel = np.sum([subdata.apply(v,axis=1).values for v in conditions],axis=0) >= min_hits
        dfi[new_col].loc[subdata.index.values[sel]] = True

    return dfi