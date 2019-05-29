import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import model_analyzer as ma
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                             RandomForestClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid

#read data
def read_csv_handledates(filepath, date_cols):
    '''
    input: filepath to a csv, list of date columns to parse
    output: a pandas dataframe with correctly typed date columns
    '''
    return pd.read_csv(filepath, parse_dates=date_cols)

#explore the data


#for creating an outcome variable and features
def calculate_duration(df, beginning_col, ending_col, new_col):
    '''
    input: a dataframe with two date columns you're interested in the time between
    output: that dataframe now has a column representing the time between them
    '''
    df[new_col] = df[ending_col] - df[beginning_col]

def compare_tdelta_generate_binary(df, col_to_compare, threshold, new_col):
    '''
    inputs: a dataseries with time deltas, a number (threshold) to compare
    members of that data series to and a name for the new column with the
    binary values
    output: that dataseries now has a column for the binary values
    '''
    df[new_col] = np.where(df[col_to_compare]<= pd.Timedelta(threshold), 1, 0)

def id_potential_features(df):
    '''
    input: dataframe
    output: list of string columns and float columns that can become features
    '''
    str_cols = [column for column in df.columns if (df[column].dtype=='O')
                   and (len(df[column].unique())<=20)]
    flt_cols = [column for column in df.columns if
                (df[column].dtype=='float64')]

    return str_cols, flt_cols

def strings_to_dummies(df, strcols):
    '''
    turn the string columns we identified into dummies and generate a new
    df with these dummies to act as features
    '''
    features = pd.get_dummies(df[strcols], dummy_na=True,
                              columns=strcols, drop_first=True)
    return features

def add_discretized_float_cols(og_df, feature_df, fltcols):
    '''
    make bins 3 or 5 please
    adds discretized float cols to a features column
    fltcols are columns from og_df to be discretized and added to feature_df
    '''
    l = ['low', 'medium low', 'medium high', 'high']
    for column in fltcols:
        new_col = column + '_d'
        feature_df[column] = og_df[column]
        feature_df[new_col] = pd.qcut(og_df[column], q=[0, .25, .5, .75, 1.],
                                    labels=l)

#clean data
def fillna_modal(df, col_list):
    '''
    fills nans in col_list with the column's mode
    '''
    for x in col_list:
        df[x].fillna(df[x].value_counts().index[0], inplace=True)

def fillzero_mean(df, col):
    '''
    needs a int or float dataseries where nans have already been turned to 0's
    '''
    imputation = df[col].notnull().mean()
    df[col].replace(0, imputation, inplace = True)

#split data (temporal and x/y)
def temp_holdout(df, date_col, tt_period, wait_period):
    '''
    big period in months, wait period in days
    wait period is holdout period after train or test period
    '''
    period = pd.DateOffset(months=tt_period)
    holdout = pd.DateOffset(days=wait_period)
    first = df[date_col].min()
    last = df[date_col].max()
    next_edge = first + period
    rv = []

    while next_edge < last:
        rv.append(df[(df[date_col] >= first) & (df[date_col] < next_edge)])
        first = next_edge + holdout
        next_edge = next_edge + period + holdout

    return rv

def seperate_ids_feats_ys(periods):
    '''
    takes a list of periods from temp_holdout() and returns a list of lists of
    each period's ids, features, and ys
    '''
    rv = []

    for x in periods:
        rv.append([x.iloc[:, [-2, -3]], x.iloc[:, :-3], x.iloc[:, -1]])
        
    for x in rv:
        for df in x:
            print(df.shape)

    return rv

#select your grid   
def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l2', C=1e5, solver='lbfgs'),
        'SVM': SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="log", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3), 
        'BAG': BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, n_estimators = 20) 
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10], 'solver': ['liblinear']},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BAG': {'n_estimators' : [5,10, 20], 'max_samples' : [.25, .5, .75]}
       }
    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1], 'solver': ['saga', 'liblinear']},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.1,0.5],'subsample' : [0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10, None],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.1],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BAG': {'n_estimators' : [5,10], 'max_samples' : [.25, .5] } 
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l2'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
    'BAG': {'n_estimators' : [5], 'max_samples' : [.25] } 

           }
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

#run and analyze models
def model_analyzer(clfs, grid, plots, x_train, y_train, x_test, y_test):
    '''
    inputs: clfs dict of default models
            selected grid
            plots ('show' to see all plots, 'save' to see save all plots)
            split training and testing data
    outputs: df of all models and their predictions/metrics
             df of all predictions with model id as column name for later use
    '''

    stats_dics = []
    predictions_dict = {}

    for klass, model in clfs.items():
        parameter_values = grid[klass]
        for p in ParameterGrid(parameter_values):
            try:
                name = klass + str(p)
                print(name)
                m = ma.ClassifierAnalyzer(model, p, name, .2, x_train, y_train,
                                          x_test, y_test)
                stats_dics.append(vars(m))
                predictions_dict[m.name] = m.predictions
                if plots == 'show':
                    m.plot_precision_recall(False, True, name + 'pr' + '.png')
                    m.plot_roc(False, True, name + 'roc')
                if plots == 'save':
                    m.plot_precision_recall(True, False, name + 'pr' + '.png')
                    m.plot_roc(True, False, name + 'pr')

            except IndexError as e:
                    print('Error:',e)
                    continue

    return stats_dics, predictions_dict

