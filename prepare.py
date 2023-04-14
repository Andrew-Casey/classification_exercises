def prep_iris():

    #Load the iris data
    iris = acq.get_iris_data()

    # Drop the unnecessary columns
    iris = iris.drop(columns=['species_id', 'measurement_id'])

    # Rename the species_name column
    iris = iris.rename(columns={'species_name': 'species'})

    # Create dummy variables for the species column
    dummy_df = pd.get_dummies(iris['species'], dummy_na=False)
    iris = pd.concat([iris, dummy_df], axis=1)

    return iris


def prep_titanic():
   
    # Load the titanic data
    titanic = acq.get_titanic_data()
    
    # Drop unnecessary columns
    titanic = titanic.drop(columns=['deck', 'embark_town', 'class', 'age'])
   
    # Create dummy variables
    dummy_cols = ['sex', 'embarked']
    dummy_df = pd.get_dummies(titanic[dummy_cols], dummy_na=False)
    titanic = pd.concat([titanic, dummy_df], axis=1)

    return titanic


def prep_telco():
    # Load the telco data
    telco = acq.get_telco_churn()
    # Drop unnecessary columns
    telco.drop(columns=['internet_service_type_id','contract_type_id','payment_type_id'])

   #Encode 
    telco['gender_encoded'] = telco['gender'].replace({'Female': 0, 'Male': 1})
    telco['senior_citizen_encoded'] = (telco['senior_citizen'] == 1).astype(int)
    telco['partner_encoded'] = telco['partner'].replace({'Yes': 1, 'No': 0})
    telco['dependents_encoded'] = telco['dependents'].replace({'Yes': 1, 'No': 0})
    telco['phone_service_encoded'] = telco['phone_service'].replace({'Yes': 1, 'No': 0})
    telco['online_security_encoded'] = telco['online_security'].replace({'Yes': 1, 'No': 0})
    telco['online_backup_encoded'] = telco['online_backup'].replace({'Yes': 1, 'No': 0})
    telco['tech_support_encoded'] = telco['tech_support'].replace({'Yes': 1, 'No': 0})
    telco['streaming_tv_encoded'] = telco['streaming_tv'].replace({'Yes': 1, 'No': 0})
    telco['streaming_movies_encoded'] = telco['streaming_movies'].replace({'Yes': 1, 'No': 0})
    telco['paperless_billing_encoded'] = telco['paperless_billing'].replace({'Yes': 1, 'No': 0})
    telco['churn_encoded'] = telco['churn'].replace({'Yes': 1, 'No': 0})
    
    #dummy variables
    dummy_cols = ['contract_type','internet_service_type','payment_type', 'multiple_lines']
    dummy_df = pd.get_dummies(telco[dummy_cols], dummy_na=False)
    telco = pd.concat([telco, dummy_df], axis=1)

    return telco

#split iris
def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train_validate.species)
    return train, validate, test

#split titanic
def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train_validate.survived)
    return train, validate, test

#split telco
def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.churn)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.churn)
    return train, validate, test


