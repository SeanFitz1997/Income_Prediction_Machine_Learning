import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

CLEAN_DATA_DIR = os.path.abspath('data/clean_data')
DATA_DIR = os.path.abspath('data')

target_encodings = {}


def get_target_mappings(df, target_column, encodeing_columns, mean_smoothing_weight=0.3):
    ''' Gets mappings for target encodings 
        df: The training data.
        target_column: The name of the target column.
        encodeing_columns: List of column names to target encode.
        mean_smoothing_weight: Target encoding are the weighted average between the
            average target value for that class and the mean target value. 
            (0 for no smoothing).
        Returns: A dictionary with the keys as column names and values as dictionaries for their amppings.
        (e.g. { 'Country' : { 'Ireland': X, ...}, ...})
    '''
    global target_encodings

    if not target_encodings:
        target_encodings[target_column] = df[target_column].mean()

        for column in encodeing_columns:
            # Create mappings
            category_mappings = ((df.groupby(column)[target_column].mean(
            ) * (1-mean_smoothing_weight)) + (target_encodings[target_column] * mean_smoothing_weight))

            # Add mappings
            target_encodings[column] = category_mappings

    return target_encodings


def get_clean_data(filepath, target_column, remove_columns=[], target_encode=[], one_hot_encode=[], standardize=True):
    ''' Cleans data
        filepath: Filepath to load data.
        target_column: The name of the target column.
        remove_columns: List of column names to remove.
        target_encode: List of column names to target encode.
        one_hot_encode: List of column names to one hot encode.
        standardize: Bool, standardizes data.
        Return: DataFrame containing cleaned data
    '''
    # Load data
    df = pd.read_csv(filepath)

    # Validate parameters
    if target_column not in df:
        raise ValueError('Target column ({}) in not a column in the dataset at ({})'.format(
            target_column, filepath))
    if target_column in remove_columns:
        raise ValueError(
            'You can\'t remove your target column ({})'.format(target_column))
    if target_column in target_encode:
        raise ValueError(
            'You can\'t endcode your target encode your target column ({})'.format(target_column))
    for column in remove_columns:
        if column in target_encode:
            raise ValueError(
                'You can\'t target encode a column you will remove ({})'.format(column))
        if column in one_hot_encode:
            raise ValueError(
                'You can\'t one hot encode a column you will remove ({})'.format(column))

    # Fix samples with missing values
    # Replace missing year & age with medians
    df['Year of Record'] = df['Year of Record'].fillna(
        df['Year of Record'].median())
    df['Age'] = df['Age'].fillna(
        df['Age'].median())
    df['Year of Record'] = df['Year of Record'].astype(int)
    df['Age'] = df['Age'].astype(int)

    # Fix gender values
    df['Gender'] = df['Gender'].replace(['0', 'unknown', np.NaN], 'other')

    # Fix university values
    df['University Degree'] = df['University Degree'].replace(
        ['0', np.NaN], 'No')

    # Fix hair color values
    df['Hair Color'] = df['Hair Color'].replace(['0', np.NaN], 'Unknown')

    # Target encode classes
    if target_encode:
        target_mappings = get_target_mappings(df, target_column, target_encode)

        for column in target_encode:
            df[column] = df[column].map(target_mappings[column]).fillna(
                target_mappings[target_column])

    # One hot encode class labels
    for column in one_hot_encode:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df.drop([column], axis=1, inplace=True)

    # Remove un-wanted features
    for column in remove_columns:
        del df[column]

    # Standardize data
    if standardize:
        scaler = StandardScaler()
        scale_columns = df.columns.to_list()
        scale_columns.remove(target_column)
        df[scale_columns] = scaler.fit_transform(df[scale_columns])

    return df


def create_submission(model, submission_data, filepath=None):
    ''' Creates a submission csv file
        model: Model used to make predictions.
        submission_data: Dataframe with cleaned submission data.
        target_variable: The name of the target column.
        filepath: The filepath to save submission
    '''
    # make predictions
    target_variable = 'Income in EUR'
    X = submission_data.drop(columns=target_variable).values
    y_pred = model.predict(X)

    # create submission file
    submission_template_filepath = os.path.realpath(
        'data\\tcd ml 2019-20 income prediction submission file.csv')

    submission_save_filepath = filepath if filepath else os.path.realpath(
        'data\\submission\\submission.csv')

    submisssion_template = pd.read_csv(submission_template_filepath)
    submisssion_template['Income'] = pd.Series(y_pred.flatten())

    submisssion_template.to_csv(submission_save_filepath, index=False)
    print(f'Submission written to {submission_save_filepath}')


def remove_outliers(X, y, z_score_threshold=20):
    ''' Removes samples with y's greater than threshold  
        X: numpy array of feature data.
        y: numpy array of target data.
        z_score_threshold
    '''
    z_scores = np.abs(stats.zscore(y))
    X = X[z_scores < z_score_threshold]
    y = y[z_scores < z_score_threshold]
    print(
        f'{len(z_scores[z_scores >= z_score_threshold])} outliers were removed.')
    return X, y


if __name__ == '__main__':
    print('--- Creating Clean Data ---')

    target_variable = 'Income in EUR'
    target_encode = ['Country', 'Profession']
    one_hot_encode = ['Gender', 'University Degree']
    remove_features = ['Instance', 'Hair Color', 'Wears Glasses']

    # Create clean training data
    training_data_filepath = os.path.join(
        DATA_DIR, 'tcd ml 2019-20 income prediction training (with labels).csv')
    df = get_clean_data(training_data_filepath, target_variable, remove_columns=remove_features,
                        target_encode=target_encode, one_hot_encode=one_hot_encode)
    clean_training_data_path = os.path.join(
        CLEAN_DATA_DIR, 'clean_training_data.csv')
    df.to_csv(clean_training_data_path)
    print('Clean training data written to: {}'.format(clean_training_data_path))

    # Create clean submission data
    submission_data_filepath = os.path.join(
        DATA_DIR, 'tcd ml 2019-20 income prediction test (without labels).csv')
    df = get_clean_data(submission_data_filepath, target_variable, remove_columns=remove_features,
                        target_encode=target_encode, one_hot_encode=one_hot_encode)
    clean_submission_data_path = os.path.join(
        CLEAN_DATA_DIR, 'clean_submission_data.csv')
    df.to_csv(clean_submission_data_path)
    print('Clean submission data written to: {}'.format(
        clean_submission_data_path))
