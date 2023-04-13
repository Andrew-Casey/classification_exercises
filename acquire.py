#imports
import numpy as np
import pandas as pd
from pydataset import data
import os



# get db url
def get_db_url(database, user=env.user, password=env.password, host_name=env.host_name):
    url = f'mysql+pymysql://{user}:{password}@{host_name}/{database}'
    return url

#get titanic
#SQL_query = 'select * from passengers'
#directory = '/Users/andrewcasey/codeup-data-science/classification_exercises'
def new_titanic_data(SQL_query):
    '''
    - This function will:
    - take in a sql query
    - create a connect_url to mySQL
    - return a df of the given query from the titanic_db
    '''
    url = get_db_url('titanic_db')
    
    return pd.read_sql(SQL_query, url)

#
def get_titanic_data(SQL_query, directory, filename='titanic.csv'):
    '''
    this function will:
    -check local directory for csv file
     -return if exists
    -if csv doesn't exist:
     -create a df of the SQL_query
     -write df to csv
    -output titanic df
    '''
    if os.path.exists(directory + filename):
        df = pd.read_csv(filename)
        return df
    else:
        df = new_titanic_data(SQL_query)
        
        #want to save to csv
        df.to_csv(filename)
        return df

# get iris
def new_iris_data(SQL_query):
    '''
    - This function will:
    - take in a sql query
    - create a connect_url to mySQL
    - return a df of the given query from the iris_db
    '''
    url = get_db_url('iris_db')
    
    return pd.read_sql(SQL_query, url)

#SQL_query = """select * 
                    #from species 
                    #join measurements using (species_id)"""
#directory = '/Users/andrewcasey/codeup-data-science/classification_exercises'

def get_iris_data(SQL_query, directory, filename='iris.csv'):
    '''
    this function will:
    -check local directory for csv file
     -return if exists
    -if csv doesn't exist:
     -create a df of the SQL_query
     -write df to csv
    -output titanic df
    '''
    if os.path.exists(directory + filename):
        df = pd.read_csv(filename)
        return df
    else:
        df = new_iris_data(SQL_query)
        
        #want to save to csv
        df.to_csv(filename)
        return df


#telco churn
def new_telco_churn(SQL_query):
    '''
    - This function will:
    - take in a sql query
    - create a connect_url to mySQL
    - return a df of the given query from the telco_churn
    '''
    url = get_db_url('telco_churn')
    
    return pd.read_sql(SQL_query, url)

def get_telco_churn(SQL_query, directory, filename='telco.csv'):
    '''
    this function will:
    -check local directory for csv file
     -return if exists
    -if csv doesn't exist:
     -create a df of the SQL_query
     -write df to csv
    -output titanic df
    '''
    if os.path.exists(directory + filename):
        df = pd.read_csv(filename)
        return df
    else:
        df = new_telco_churn(SQL_query)
        
        #want to save to csv
        df.to_csv(filename)
        return df

# SQL_query = """select * 
                    #from customers
                    #join contract_types using (contract_type_id)
                    #join internet_service_types using (internet_service_type_id)
                    #join payment_types using (payment_type_id)"""
#directory = '/Users/andrewcasey/codeup-data-science/classification_exercises'