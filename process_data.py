# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder #, OneHotEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


################################################
########### IMPORTING FILE #####################
################################################

train = pd.read_csv('Train.csv', parse_dates=['disbursement_date','due_date'])
test = pd.read_csv('Test.csv', parse_dates=['disbursement_date','due_date'])

test['target'] = -1
df = pd.concat([train, test], ignore_index=True).reset_index(drop=True)

################################################
########### PREPROCESS DATA ####################
################################################

print("Initializing Data Processing.")
df['duration'] = (df['duration']/7).round(0).astype('int64')
# df['duration'] = df['duration'].astype('category')

col_to_int = ['Total_Amount','Total_Amount_to_Repay', 'Amount_Funded_By_Lender','Lender_portion_to_be_repaid']
df[col_to_int] = df[col_to_int].round(0).astype('int64')
df['year'] = df['disbursement_date'].dt.year

ordinal = OrdinalEncoder(dtype='int')
# onehot = OneHotEncoder(sparse_output=False, dtype='int')
cats = ['lender_id', 'loan_type', 'year']
for cat in cats:
    df[cat] = ordinal.fit_transform(df[[cat]])

todrop = ['disbursement_date', 'due_date', 'country_id', 'tbl_loan_id', 'New_versus_Repeat']
df = df.drop(todrop, axis=1)
df['duration'] = df['duration'].astype('int')
train = df[df['target'] != -1]
test = df[df['target'] == -1]
train.to_csv('./data/cleaned_train.csv', index=False)
test.to_csv('./data/cleaned_test.csv', index=False)
