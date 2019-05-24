import numpy as np
import pandas as pd
import model_analyzer as ma
from sklearn.dummy import DummyClassifier

#read data
def read_csv_handledates(filepath, date_cols):
    '''
    input: filepath to a csv, list of date columns to parse
    output: a pandas dataframe with correctly typed date columns
    '''
    return pd.read_csv(filepath, parse_dates=date_cols)

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

def discretize_float_cols(og_df, feature_df, fltcols, b):
    '''
    make bins 3 or 5 please
    adds discretized float cols to a features column
    fltcols are columns from og_df to be discretized and added to feature_df
    '''
    if b == 3:
        l = ['low', 'medium', 'high']
    if b == 5:
        l = ['very low', 'low', 'medium', 'high', 'very high']
    for column in fltcols:
        feature_df[column] = pd.cut(og_df[column], bins=b,
                                    labels=l)
        
#clean data

#split data (temporal and x/y)

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
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3), 
        'BAG': BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, n_estimators = 20) 
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
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
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.1,0.5],'subsample' : [0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.1],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BAG': {'n_estimators' : [5,10], 'max_samples' : [.25, .5] } 
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
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
def model_analyzer(clfs, grid, x_train, y_train, x_test, y_test):
    '''
    inputs: clfs dict of default models
            selected grid
            plots ('show' to see all plots, 'save' to see and save all plots)
            split training and testing data
    outputs: df of all models and their predictions/metrics
             df of all predictions with model id as column name for later use
    '''

    stats_df = pd.DataFrame()
    predictions_dict = {}

    for klass, model in clfs.items():
        parameter_values = grid[klass]
        for p in ParameterGrid(parameter_values):
            try:
                name = klass + str(p)
                m = ma.ClassifierAnalyzer(model, p, .2, x_train, y_train,
                                          x_test, y_test)
                result_df.concat(pd.DataFrame(vars(m)))
                predictions_dict[m] = m.predictions

            except IndexError as e:
                    print('Error:',e)
                    continue

    return stats_df, pd.DataFrame(predictions_dict)