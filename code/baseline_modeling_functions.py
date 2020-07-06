import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

def split_data(df, target_variable, **kwargs):
    
    '''
    Using train_test_split with stratify to keep class representation in the data the same across training
    and testing data.
    '''

    if kwargs:
        kwarg_list = list(kwargs.items())
        df = df[(df[kwarg_list[0][0]] == kwarg_list[0][1]) & 
                (df[kwarg_list[1][0]] == kwarg_list[1][1])]
    
    X = df.drop([target_variable, 'ab_id', 'batter_id', 'g_id', 'pitcher_id', 'px', 'pz',
                 'CH', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'FT', 'IN',
                 'KC', 'KN', 'PO', 'SC', 'SI', 'SL', 'UN', 'id', 'year', 'target'], axis=1).select_dtypes(exclude='object')
    y = df[target_variable]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=47, stratify=y)

    return X_tr, X_te, y_tr, y_te

def find_best_params(df, target_variable, **kwargs):
    
    """
    Finding the best parameters of the models using cross validation to find the most accurate models
    """
    
    X_train, X_test, y_train, y_test = split_data(df, target_variable, **kwargs)
    
    best_C_list = []
    best_alpha_list = []
    model_list = [LogisticRegression(), RidgeClassifier()]
    
    for model in model_list:
        if model == model_list[0]:
            steps = [('clf', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000, random_state=11))]
            param_grid = {'clf__C': [.0001, .001, .01, .1]}

            pipeline = Pipeline(steps)
            cv = GridSearchCV(pipeline, param_grid, cv=3)
            cv.fit(X_train, y_train)

            best_C = cv.best_params_['clf__C']
            best_C_list.append(best_C)

        elif model == model_list[1]:
            steps = [('ridge', RidgeClassifier(random_state=11))]
            param_grid = {'ridge__alpha': np.arange(10,100,10)}

            pipeline = Pipeline(steps)
            cv = GridSearchCV(pipeline, param_grid, cv=3)
            cv.fit(X_train, y_train)

            best_alpha = cv.best_params_['ridge__alpha']
            best_alpha_list.append(best_alpha)  
                    
    best_params_df = pd.DataFrame({'Logistic Regression': best_C_list, 'Ridge Classifier': best_alpha_list})
            
    return best_params_df

def get_model_accuracies(df, target_variable, **kwargs):
    
    '''
    Gives a data frame featuring the accuracies of the models
    '''
    
    best_params_df = find_best_params(df, target_variable, **kwargs)
    
    X_train, X_test, y_train, y_test = split_data(df, target_variable, **kwargs)

    best_C = best_params_df.iloc[0]['Logistic Regression']
    best_alpha = best_params_df.iloc[0]['Ridge Classifier']

    clf = LogisticRegression(C=best_C, solver='lbfgs', multi_class='auto', max_iter=10000, random_state=11).fit(X_train, y_train)
    rlf = RidgeClassifier(alpha=best_alpha, random_state=11).fit(X_train, y_train)

    clf_training_score = clf.score(X_train, y_train)
    clf_testing_score = clf.score(X_test, y_test)
    rlf_training_score = rlf.score(X_train, y_train)
    rlf_testing_score = rlf.score(X_test, y_test)
    
    scores_df = pd.DataFrame({'Logistic Regression Training Scores': clf_training_score, 
                              'Logistic Regression Testing Scores': clf_testing_score,
                              'Ridge Classifier Training Scores': rlf_training_score,
                              'Ridge Classifier Testing Scores': rlf_testing_score},
                              index=['combined_df'])
    scores_df = scores_df.transpose()
    
    return scores_df

def get_model_stats(df, target_variable, **kwargs):
    
    '''
    Gives confusion matrices and classification reports for the models in use
    '''
    
    X_train, X_test, y_train, y_test = split_data(df, target_variable, **kwargs)
    
    best_params_df = find_best_params(df, target_variable, **kwargs)
    
    best_C = best_params_df.iloc[0]['Logistic Regression']
    best_alpha = best_params_df.iloc[0]['Ridge Classifier']

    logit = LogisticRegression(C=best_C, solver='lbfgs', multi_class='auto', max_iter=10000, random_state=11)
    ridge = RidgeClassifier(alpha=best_alpha, random_state=11)

    clf = logit.fit(X_train, y_train)
    rlf = ridge.fit(X_train, y_train)
    
    logit_predict = logit.predict(X_test)
    ridge_predict = ridge.predict(X_test)

    lcm = confusion_matrix(logit_predict, y_test)
    rcm = confusion_matrix(ridge_predict, y_test)

    plt.figure(figsize=(9,9))
    sns.heatmap(lcm, annot=True, fmt='.0f', linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Logistic Regression Accuracy Score: {0}'.format(round(logit.score(X_test, y_test) * 100, 2))
    plt.title(all_sample_title, size = 15)
    plt.show()

    plt.figure(figsize=(9,9))
    sns.heatmap(rcm, annot=True, fmt='.0f', linewidths=.5, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Ridge Classifier Accuracy Score: {0}'.format(round(ridge.score(X_test, y_test) * 100, 2))
    plt.title(all_sample_title, size = 15)
    plt.show()
    
    print('Logistic Regression Classification Report')
    print(classification_report(y_test, logit_predict))
    print('Ridge Classifier Classification Report')
    print(classification_report(y_test, ridge_predict))