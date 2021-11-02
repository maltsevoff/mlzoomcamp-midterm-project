import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# parameters

C = 0.5
n_splits = 5
output_file = f'model_C={C}.bin'

# data preparation

df = pd.read_csv('data.csv')

columns_to_rename = {
    'V1': 'age',
    'V2': 'job',
    'V3': 'marital',
    'V4': 'education',
    'V5': 'default',
    'V6': 'balance',
    'V7': 'housing',
    'V8': 'loan',
    'V9': 'contact',
    'V10': 'day',
    'V11': 'month',
    'V12': 'duration',
    'V13': 'campaing',
    'V14': 'pdays',
    'V15': 'previous',
    'V16': 'poutcome',
    'Class': 'subscribe'
}

df = df.rename(columns=columns_to_rename)

df.subscribe = df.subscribe - 1

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

numerical = ['duration']
categorical = ['job', 'month', 'poutcome', 'housing']

# training 

def train(df_train, y_train, C=1.0):
    dv = DictVectorizer(sparse=False)

    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model):
    val_dict = df[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    y_pred = model.predict_proba(X_val)[:, 1]
    return y_pred


# validation

print(f'doing validation with C={C}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.subscribe.values
    y_val = df_val.subscribe.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')

    fold = fold + 1

print('validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# training the final model 

dv, model = train(df_full_train, df_full_train.subscribe.values, C=C)
y_pred = predict(df_test, dv, model)

y_test = df_test.subscribe.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

# save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')