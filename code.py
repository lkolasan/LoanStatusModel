import pandas as pd

pd.options.mode.chained_assignment = None

original_data = pd.read_csv("LendingClub2012to2013.csv", low_memory = False, skiprows=[0])
original_data.shape

print(original_data['loan_status'].unique())
print(original_data.shape)

no_incomplete_rows = original_data[original_data['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])]
no_incomplete_rows['loan_status'] = no_incomplete_rows['loan_status'].apply(lambda x: 0 if x == "Fully Paid" else 1)
no_incomplete_rows.shape

no_incomplete_rows.to_csv("test.csv", index = False)

customer_id = ['id']
leakage_to_drop = ['recoveries', 'collection_recovery_fee', 'last_fico_range_high', 'last_fico_range_low', 'last_credit_pull_d', 'total_rec_prncp', 'last_pymnt_amnt', 'total_pymnt', 'total_pymnt_inv', 'last_pymnt_d', 'total_rec_late_fee', 'total_rec_int', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'next_pymnt_d']

no_information_features_to_drop = ['all_util', 'dti_joint', 'il_util', 'inq_fi', 'inq_last_12m', 'max_bal_bc', 'mths_since_rcnt_il', 'open_acc_6m', 'open_il_12m', 'open_il_24m', 'open_il_6m', 'open_rv_12m', 'open_rv_24m', 'total_bal_il', 'verification_status_joint', 'annual_inc_joint', 'application_type',  'policy_code', 'total_cu_tl']

no_leakage = no_incomplete_rows.drop(leakage_to_drop + no_information_features_to_drop + customer_id, axis = 1)

no_leakage.to_csv("test.csv", index = False)

import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
from sklearn import linear_model

plotting_data = no_incomplete_rows[['last_fico_range_low', 'loan_status']]
plotting_data = plotting_data[plotting_data['last_fico_range_low']>50]
x = plotting_data['last_fico_range_low']
y = plotting_data['loan_status']

model = linear_model.LogisticRegression()
model.fit(x.values.reshape(-1, 1),y)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

x_test = np.linspace(x.min(), x.max(), num=500)
y_test = pd.DataFrame(model.predict_proba(x_test.reshape(-1,1)), columns=['zero', 'one'])['one']

plt.scatter(x, y, alpha=0.002, marker='o')
plt.scatter(x_test, y_test, marker='.', color = 'red')

from sklearn import preprocessing
from sklearn import model_selection
import numpy as np

variables_i_want_to_try = ['int_rate', 'grade', 'term', 'fico_range_high']
df = no_leakage[variables_i_want_to_try]

term_dummies = pd.get_dummies(df['term'], prefix="term_")
grade_dummies = pd.get_dummies(df['grade'], prefix="grade")

df = pd.concat([df, term_dummies, grade_dummies], axis=1)

df.rename(columns={'term__ 36 months': 'term_36_months', 'term__ 60 months': 'term_60_months'}, inplace=True)

df['int_rate'] = df['int_rate'].apply(lambda x: x.strip('%'))

variables_with_dummies = ['int_rate', 'term_36_months', 'term_60_months','fico_range_high', 'grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G']

X = df[variables_with_dummies]
y = no_leakage['loan_status']

cv = model_selection.KFold(5)
lr = linear_model.LogisticRegression(C=1000000000000)

included = []
score_values = [0.5]
while len(included) < len(variables_with_dummies):
    score = 0.5
    best = ""
    for i in variables_with_dummies:
        if i not in included:
            attempt = included + [i]
            new_score = np.mean(model_selection.cross_val_score(lr, X[attempt], y, cv=cv, scoring='roc_auc'))
            print("     %s: %f" %(str(attempt), new_score))
            if new_score > score: 
                best = i
                score = new_score
    included = included + [best]
    score_values = score_values + [score]
    print("Best model for this round: %s: %f" %(str(included), score))
    
plt.plot(score_values)
plt.ylabel('Model Scores')
plt.show()
