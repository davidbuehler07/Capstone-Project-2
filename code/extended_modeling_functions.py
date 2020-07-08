import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

def sample(df, target_variable, sample_df=False, sample_size=100):
    
    '''
    Pulls a user defined sample of the given data set, and keeps the class representation of the original data set.
    '''
    
    df.drop(df[df[target_variable] == 'OT'].index, inplace=True)
    appended_data = []
    perc_dict = {k:v/len(df) for k, v in list(df[target_variable].value_counts().items())[0:3]}
    
    for k in perc_dict.keys():
        n = round(sample_size * perc_dict[k])
        data = df[df[target_variable] == k].sample(n=n, random_state=11)
        appended_data.append(data)

    sample_df = pd.concat(appended_data)
    
    return sample_df

def split_data(df, target_variable, sample_df=False, sample_size=100):
    
    '''
    Function that does the train_test_split. Can sample the data set before hand, 
    or choose a specific pitcher by specifying first_name and last_name
    '''

    if sample_df:
        df = sample(df, target_variable, sample_df, sample_size)

    X = df.drop([target_variable, 'ab_id', 'batter_id', 'g_id', 'pitcher_id', 'px', 'pz',
                 'CH', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'FT', 'IN',
                 'KC', 'KN', 'PO', 'SC', 'SI', 'SL', 'UN', 'id', 'year', 'target'], axis=1).select_dtypes(exclude='object')
    y = df[target_variable]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=47, stratify=y)
        
    return X_tr, X_te, y_tr, y_te

def find_best_params(df, target_variable, sample_df=False, sample_size=100):
    
    '''
    Finds the best parameters of each individual model via cross validation for future modeling use.
    '''
    
    X_train, X_test, y_train, y_test = split_data(df, target_variable, sample_df, sample_size)
    
    model_list = [RandomForestClassifier(), GradientBoostingClassifier(), 
                  AdaBoostClassifier(), KNeighborsClassifier()]
        
    for model in model_list:
        if model == model_list[0]:
            steps = [('rf', RandomForestClassifier(random_state=11))]
            param_grid = {'rf__n_estimators': np.arange(500, 5000, 500)}

            pipeline = Pipeline(steps)
            cv = GridSearchCV(pipeline, param_grid, cv=3)
            cv.fit(X_train, y_train)

            best_rf = cv.best_params_['rf__n_estimators']
            
        elif model == model_list[1]:
            steps = [('gboost', GradientBoostingClassifier(random_state=11))]
            param_grid = {'gboost__n_estimators': np.arange(500, 1500, 500)}

            pipeline = Pipeline(steps)
            cv = GridSearchCV(pipeline, param_grid, cv=3)
            cv.fit(X_train, y_train)

            best_gboost = cv.best_params_['gboost__n_estimators']
                        
        elif model == model_list[2]:
            steps = [('aboost', AdaBoostClassifier(random_state=11))]
            param_grid = {'aboost__n_estimators': np.arange(500, 5000, 500)}

            pipeline = Pipeline(steps)
            cv = GridSearchCV(pipeline, param_grid, cv=3)
            cv.fit(X_train, y_train)

            best_aboost = cv.best_params_['aboost__n_estimators']
                        
        elif model == model_list[3]:
            steps = [('knn', KNeighborsClassifier())]
            param_grid = {'knn__n_neighbors': np.arange(9, 23, 2)}

            pipeline = Pipeline(steps)
            cv = GridSearchCV(pipeline, param_grid, cv=3)
            cv.fit(X_train, y_train)

            best_knn = cv.best_params_['knn__n_neighbors']
        
    return best_rf, best_gboost, best_aboost, best_knn

def get_classification_report(df, target_variable, sample_df=False, sample_size=100):
    
    '''
    All chosen models get input from the find_best_params function, 
    and is used to fit and predict the data
    '''
    
    X_train, X_test, y_train, y_test = split_data(df, target_variable, sample_df, sample_size)
    
    best_rf, best_gboost, best_aboost, best_knn = find_best_params(df, target_variable, sample_df, sample_size)
    
    rf = RandomForestClassifier(n_estimators=best_rf, random_state=11)
    gboost = GradientBoostingClassifier(n_estimators=best_gboost, random_state=11)
    aboost = AdaBoostClassifier(n_estimators=best_aboost, random_state=11)
    knn = KNeighborsClassifier(n_neighbors=best_knn)
    
    rf_fit = rf.fit(X_train, y_train)
    gboost_fit = gboost.fit(X_train, y_train)
    aboost_fit = aboost.fit(X_train, y_train)
    knn_fit = knn.fit(X_train, y_train)
    
    rf_predict = rf.predict(X_test)
    gboost_predict = gboost.predict(X_test)
    aboost_predict = aboost.predict(X_test)
    knn_predict = knn.predict(X_test)

    target_names = rf.classes_
    print('Random Forest Classification Report')
    print(classification_report(y_test, rf_predict, target_names=target_names))
    print('Gradient Boosting Classification Report')
    print(classification_report(y_test, gboost_predict, target_names=target_names))
    print('Ada Boosting Classification Report')
    print(classification_report(y_test, aboost_predict, target_names=target_names))
    print('K Neighbors Classification Report')
    print(classification_report(y_test, knn_predict, target_names=target_names))
    
def gradient_boost(df, target_variable, resample=False, bb_mult=.75, os_mult=.5, sample_df=False, sample_size=100):
    
    '''
    Gradient boosting won as the best model. This function specifically focuses on gradient boosting and 
    gives feature importances in a graphical form.
    '''

    X_train, X_test, y_train, y_test = split_data(df, target_variable, sample_df, sample_size)
    
    if resample:
        sm = SMOTE(sampling_strategy= {'FB': y_train.value_counts().values[0],
                                       'BB': int(y_train.value_counts().values[0] * bb_mult),
                                       'OS': int(y_train.value_counts().values[0] * os_mult)})
        X_train, y_train = sm.fit_resample(X_train, y_train)

    steps = [('gboost', GradientBoostingClassifier(random_state=11))]
    param_grid = {'gboost__n_estimators': np.arange(100, 1100, 100)}

    pipeline = Pipeline(steps)
    cv = GridSearchCV(pipeline, param_grid, cv=3)
    cv.fit(X_train, y_train)

    best_gboost = cv.best_params_['gboost__n_estimators']

    gboost = GradientBoostingClassifier(n_estimators=best_gboost, random_state=11)
    gboost_fit = gboost.fit(X_train, y_train)
    gboost_predict = gboost.predict(X_test)

    feature_importances = pd.DataFrame(pd.Series(gboost.feature_importances_), columns=['Importance'])
    feature_importances.index = X_train.columns
    feature_importances = feature_importances.sort_values('Importance')

    plt.figure(figsize=(18,10))
    plt.barh(feature_importances.index, width=feature_importances['Importance'] * 100, color='r')
    plt.title("Gradient Boosting Feature Importances (%)")
    plt.xlabel('Percentage Importance')
    plt.show()
    
    print(f'Gradient boost train accuracy:  {round(gboost.score(X_train, y_train) * 100, 2)}%')
    print(f'Gradient boost test accuracy:   {round(gboost.score(X_test, y_test) * 100, 2)}%')
    print('\n')
    
    target_names = gboost.classes_
    print('Gradient Boosting Classification Report')
    print(classification_report(y_test, gboost_predict, target_names=target_names))