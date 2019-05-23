'''
HW 3

Dave Foote
''' 

'''
Explore and Pre-Process the Data
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pipe_lib as pl
import datetime
import calendar

def donation_dateprep(df):
    '''
    combine several datetime processing steps into one function to improve
    readability of jupyter notebook
    '''
    df['datefullyfunded'] = convert_with_format(df, 'datefullyfunded')
    df['date_posted'] = convert_with_format(df, 'date_posted')
    dys = list(df['datefullyfunded'] - df['date_posted'])
    df['days_to_funding'] = [d.days for d in dys]
    df['Y'] = np.where(df['days_to_funding'] >= 60, 1, 0)
    
def rolling_window_splitter(df, date_col, window, features):
    '''
    splits df into 6 month periods based on a column
    window is in months
    '''
    features.append('Y')
    features.append('date_posted')
    df = df.sort_values('date_posted')
    df = df.loc[:,features]
    start = pd.Timestamp(df.iloc[0][date_col])
    next_edge = pd.Timestamp(add_months(start, window))
    end = pd.Timestamp(df.iloc[-1][date_col])
    rv = []
    
    while next_edge <= end:
        rv.append(df.loc[(df[date_col] < next_edge) & (df[date_col] > start)])
        start = next_edge
        next_edge = pd.Timestamp(add_months(start, window))
        
    rv.append(df.loc[df[date_col] > start])
    features.pop()
    features.pop()
        
    return rv

def x_y_split(data):
    y = data.Y
    x = data.drop('Y', axis=1)
    return x, y

def convert_with_format(df, column_name):
    return pd.to_datetime(df[column_name], format='%m/%d/%y')
    
def add_months(start, months):
    '''
    sourced from stack overflow:
    https://stackoverflow.com/questions/4130922/how-to-increment-datetime-by-custom-months-in-python-without-using-library
    '''
    month = start.month - 1 + months
    year = start.year + month // 12
    month = month % 12 + 1
    day = min(start.day, calendar.monthrange(year,month)[1])
    
    return datetime.date(year, month, day)
    
def summarize_donations(df):
    '''
    some light data exploration for the donations
    '''
    print('percent of all projects that get funded in 60 days: ', df.Y.mean())
    print('average days it takes for a project to get funded: ',
          df.days_to_funding.mean())
    print('rate of funding within 60 days by poverty level: ',
         df.groupby('poverty_level').Y.mean())
    print('average days to funding by poverty level: ',
         df.groupby('poverty_level').days_to_funding.mean())
    print('rate of funding within 60 days by primary focus: ',
         df.groupby('primary_focus_area').Y.mean())
    print('average days to funding by primary focus: ',
         df.groupby('primary_focus_area').days_to_funding.mean())
    print('rate of funding within 60 days by state: ',
         df.groupby('school_state').Y.mean())
    print('average days to funding by state: ',
          df.groupby('school_state').Y.mean())

    