# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')


################################################
########### IMPORTING FILE #####################
################################################

train = pd.read_csv('./data/cleaned_train.csv')
print(f"Train Shape: {train.shape[0]} rows and {train.shape[1]} columns")


################################################
######### FEATURES CREATION. ###################
################################################

train['loan_interest'] = train['Total_Amount_to_Repay'] - train['Total_Amount']
train['loan_rate'] = train['loan_interest']/ train['Total_Amount']
train['lender_interest'] = train['Lender_portion_to_be_repaid'] - train['Amount_Funded_By_Lender']
# train['year'] = train['disbursement_date'].dt.year

#################################################
################### MODELLING.###################
#################################################

todrop = ["ID", 'customer_id', 'target', 'lender_id', 'loan_type', 'Total_Amount_to_Repay',
        'Lender_portion_to_be_repaid', 'Amount_Funded_By_Lender',
        ]

print("Initializing modelling")
trainx, val = train_test_split(train, stratify=train['target'], random_state=23)
X_train = trainx.drop(todrop, axis=1)
y_train = trainx.target
X_val = val.drop(todrop, axis=1)
y_val = val.target

print(f"Features use during training are: {list(X_train.column)}")
lgbclf = lgb.LGBMClassifier(random_state=23, force_col_wise=True, max_depth=11, n_estimators=500,
                            importance_type='gain', #early_stopping_rounds=100
                            )
lgbclf.fit(X_train, y_train, categorical_feature='from_dtype', eval_set=(X_val, y_val), eval_metric='F1',
           )

# make prediction and evaluate model.
y_pred = lgbclf.predict(X_val)

train_f1 = f1_score(y_train, lgbclf.predict(X_train), average='binary')
train_roc_auc = roc_auc_score(y_train, lgbclf.predict_proba(X_train)[:,1])

val_f1 = f1_score(y_val, y_pred, average='binary')
val_roc_auc = roc_auc_score(y_val, lgbclf.predict_proba(X_val)[:,1])

# write metrics to a file.
with open('metrics.txt', 'w') as outfile:
    outfile.write("Training F1 score: %.3f%%\n" %train_f1)
    outfile.write("Training ROC AUC score: %.2f%%\n" %train_roc_auc)
    outfile.write("Validation F1 score: %.3f%%\n" %val_f1)
    outfile.write("Validation ROC AUC score: %.3f%%\n" %val_roc_auc)

################################################
########## PLOT FEATURE IMPORTANCE.#############
################################################

importances = lgbclf.feature_importances_
labels = X_train.columns

feature_df = pd.DataFrame(list(zip(labels, importances)), columns=['features', 'importances'])
feature_df = feature_df.sort_values(by='importances', ascending=False)
# image formatting.
axis_fs = 18 #axis font size
title_fs = 22 #title font size
sns.set(style='whitegrid')


ax = sns.barplot(data=feature_df, y='features', x='importances')
ax.set_xlabel("Importance", fontsize=axis_fs)
ax.set_ylabel('Feature', fontsize=axis_fs)
ax.set_title('LightGBM\n feature importance', fontsize=title_fs)
plt.tight_layout()
plt.savefig("./feature_importance.png", dpi=120)
plt.close()

################################################
########## PLOT CONFUSION MATRIX. ##############
################################################
ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
plt.title("Model confusion Matrix")
plt.savefig('./confusion_matrix.png', dpi=120)
plt.close()

###############################################
########### CLASSIFICATION REPORT #############
###############################################

class_report = classification_report(y_val, y_pred)
print(class_report)
print("Training Done..")
