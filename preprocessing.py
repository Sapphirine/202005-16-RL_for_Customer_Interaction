import numpy as np
import pandas as pd
import datetime
from calendar import monthrange
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def preprocess(raw_df, episode,episode_length):
    prepared_df, rewards = prepare_actions(raw_df,episode,episode_length)

    pre_reward = prepared_df[["CONTROLN","reward"]].groupby("CONTROLN").sum()

    states = prepare_states(raw_df)
    features = random_forest(states.drop(columns=['CONTROLN']), pre_reward, 50, 0.05)
    features = np.append(features, "CONTROLN")
    states = states[features]

    merged=pd.merge(prepared_df[["CONTROLN","action","rfm","reward",'key', 'decision']],states, how='left', on="CONTROLN")

    features = random_forest(merged.drop(columns=['CONTROLN','action','reward','key','decision']),merged['reward'], 30, 0.05)
    features=np.append(features,['CONTROLN','action','reward','key', 'decision'])
    merged = merged[features]

    merged["reward"] = merged[["reward"]].groupby(merged["CONTROLN"]).cumsum()
    merged = pd.get_dummies(merged)

    X = merged.drop(columns=['CONTROLN','reward','key']).values
    y = merged["reward"].values
    X = np.array_split(X, episode)
    y = np.array_split(y, episode)
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y, rewards

def prepare_actions(raw_df,episode,episode_length):
    def reward(row):
        if pd.isnull(row["action_date"]):
            return 0
        if pd.isnull(row["reward_date"]):
            return -0.68
        else:
            return (row["reward"] - 0.68)

    def action(row):
        if pd.isnull(row["action_date"]):
            return "NA"
        else:
            return row["action"]

    def decision(row):
        if pd.isnull(row["action_date"]):
            return 0
        else:
            return 1

    action_mapping = {
        2: '97NK',
        3: '96NK',
        4: '96TK',
        5: '96SK',
        6: '96LL',
        7: '96G1',
        8: '96GK',
        9: '96CC',
        10: '96WL',
        11: '96X1',
        12: '96XK',
        13: '95FS',
        14: '95NK',
        15: '95TK',
        16: '95LL',
        17: '95G1',
        18: '95GK',
        19: '95CC',
        20: '95WL',
        21: '95X1',
        22: '95XK',
        23: '94FS',
        24: '94NK'
    }

    list_df = []
    for i in tqdm(range(2, 2+episode_length)):
        act_date = 'ADATE_{}'.format(i)
        RFM = 'RFA_{}'.format(i)

        if (i == 2):
            rew_date = 'RDATE_3'
            rew_amt = 'TARGET_D'
        else:
            rew_date = 'RDATE_{}'.format(i)
            rew_amt = 'RAMNT_{}'.format(i)

        tmp = raw_df[['CONTROLN', act_date, rew_date, rew_amt, RFM]]
        tmp = tmp.rename(columns={act_date: 'action_date', rew_date: 'reward_date', rew_amt: 'reward', RFM: 'rfm'})
        tmp['action'] = action_mapping[i]
        tmp['key'] = i
        list_df.append(tmp)

    prepared_df = pd.concat(list_df)
    prepared_df = prepared_df.sort_values(['CONTROLN', 'key'])
    prepared_df = prepared_df.reset_index().drop(columns=["index"])

    prepared_df["reward"] = prepared_df.apply(reward, axis=1)
    prepared_df["action"] = prepared_df.apply(action, axis=1)
    prepared_df["rfm"] = prepared_df["rfm"].apply(lambda x: np.NaN if x == ' ' else x)
    prepared_df["rfm"] = prepared_df[["CONTROLN", "rfm"]].groupby("CONTROLN").fillna(method='bfill')['rfm']
    prepared_df["rfm"] = prepared_df[["CONTROLN", "rfm"]].groupby("CONTROLN").fillna(method='ffill')['rfm']
    prepared_df["decision"] = prepared_df.apply(decision, axis=1)

    rewards = prepared_df['reward'].values
    s_true = prepared_df['reward'].sum()

    return prepared_df, rewards, s_true



def prepare_states(raw_df):
    state_cols = [
        'CONTROLN',
        'AGE',
        'HOMEOWNR',
        'NUMCHLD',
        'INCOME',
        'GENDER',
        'WEALTH1',
        'MAJOR',
        'WEALTH2',
        'BIBLE',
        'CATLG',
        'HOMEE',
        'PETS',
        'CDPLAY',
        'STEREO',
        'PCOWNERS',
        'PHOTO',
        'CRAFTS',
        'FISHER',
        'GARDENIN',
        'BOATS',
        'WALKER',
        'KIDSTUFF',
        'CARDS',
        'PLATES',
        'CLUSTER',
        'CLUSTER2',
        'RAMNTALL',
        'NGIFTALL',
        'CARDGIFT',
        'AVGGIFT',
        'NUMPROM'

    ]

    states = raw_df[state_cols]
    min_max_scaler = preprocessing.MinMaxScaler()

    for c in state_cols:
        if (c != 'CONTROLN') & (states[c].dtype != np.dtype('O')):
            states[[c]] = min_max_scaler.fit_transform(states[[c]])

    nan_columns = states.columns[states.isna().any()]

    imputedValues = {}
    for c in nan_columns:
        if states[c].dtype == np.dtype('O'):
            imputedValues[c] = 'NA'
        else:
            imputedValues[c] = 0
    states = states.fillna(imputedValues)

    l = []
    for c in states.columns:
        if states[c].dtype == 'O':
            l.append(c)
    for i in ["HOMEOWNR", "GENDER", "CLUSTER"]:
        l.remove(i)
    states = states.drop(columns=l)

    states = pd.get_dummies(states)

    return states


def random_forest(X, y, n_estimators, importance):
    modeloRF = RandomForestRegressor(bootstrap=False,
                                     max_features=0.3,
                                     min_samples_leaf=15,
                                     min_samples_split=8,
                                     n_estimators=n_estimators,
                                     n_jobs=-1,
                                     random_state=42)
    modeloRF.fit(X, y)

    feature_importance_df = pd.DataFrame(X.columns, columns=['Feature'])
    feature_importance_df['importance'] = pd.DataFrame(modeloRF.feature_importances_.astype(float))

    result = feature_importance_df.sort_values('importance', ascending=False)

    features = result[result.iloc[:, 1] > importance]['Feature'].values

    return features

