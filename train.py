import os
import time
import datetime
import json
import pickle
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from data import CLEAN_DATA_DIR, remove_outliers, create_submission
from visualize import view_feature_distribution, view_feature_boxplot

MODELS_DIR = os.path.abspath('models')
SPLIT_SEED = 21


def create_nn_model(input_size):
    ''' Creates Neural Network (128 -> 128 -> 1)
        input_size: The number of training features.
        Returns: NN Model 
    '''
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=[input_size]),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1)
    ])
    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=[
                  keras.metrics.RootMeanSquaredError(name='rmse')])
    return model


def train_model(model, X, y):
    ''' Trains Model
        model: Model to train
        X: Feature data
        y: Target data
    '''
    if isinstance(model, keras.Sequential):
        # Train tensorflow model
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15)
        EPOCHS = 1000
        model.fit(X, y, epochs=EPOCHS, validation_split=0.2,
                  callbacks=[early_stop])
    else:
        # Train sklearn model
        model.fit(X, y)


def test_model(model, X, y):
    ''' Test model
        model: model to test
        X: test feature data
        y: test targets
        Returns: Accuracy of predictions (RMSE), numpy array of predictions
    '''
    y_pred = model.predict(X)
    accuracy = sqrt(mean_squared_error(y, y_pred))
    return accuracy, y_pred


def archive_model(name, model, predictions, performances, training_time, submission_data):
    ''' Achieves trained model information on the model and its submission.
        name: String name of model.
        model: The trained model to predict and save.
        predictions: Model predictions for the train, test and full datasets (list of arrays)
        performances: Model performances for the training, test and full datasets (list of RMSE values)
        training_time: Time took to train
        submission_data: Cleaned submission data.
    '''
    # Create model dir
    archive_date = str(datetime.date.today())
    dir_name = f'{name}_train{int(performances[0])}_test{int(performances[1])}_full{int(performances[2])}_{archive_date}'
    archive_model_dir = os.path.join(MODELS_DIR, dir_name)
    if not os.path.exists(archive_model_dir):
        os.mkdir(archive_model_dir)

    # Record performance info
    with open(os.path.join(archive_model_dir, 'performance.txt'), 'w') as performance_file:
        performance_file.write(f'Train: {performances[0]}\n')
        performance_file.write(f'Test: {performances[1]}\n')
        performance_file.write(f'Full: {performances[2]}\n')

    # Record parameter info
    with open(os.path.join(archive_model_dir, 'parameters.json'), 'w') as parameters_file:
        if isinstance(model, keras.Sequential):
            model.summary(print_fn=lambda x: parameters_file.write(x))
        else:
            parameters = model.get_params()
            json.dump(parameters, parameters_file)

    # Record predictions
    pd.DataFrame(predictions[0]).to_csv(
        os.path.join(archive_model_dir, 'train_pred.csv'))
    pd.DataFrame(predictions[1]).to_csv(
        os.path.join(archive_model_dir, 'test_pred.csv'))
    pd.DataFrame(predictions[2]).to_csv(
        os.path.join(archive_model_dir, 'full_pred.csv'))

    # Create submission using model
    create_submission(model, submission_data, os.path.join(
        archive_model_dir, 'submission.csv'))

    # Store model object
    if isinstance(model, keras.Sequential):
        model.save(os.path.join(archive_model_dir, 'model.h5'))
    else:
        with open(os.path.join(archive_model_dir, 'model.p'), 'wb') as model_file:
            pickle.dump(model, model_file)

    print(f'model {name} archived to {archive_model_dir}')


if __name__ == '__main__':
    ''' 1. Load and split data '''
    # Get data
    target_variable = 'Income in EUR'
    clean_training_data_path = os.path.join(
        CLEAN_DATA_DIR, 'clean_training_data.csv')
    clean_submission_data_path = os.path.join(
        CLEAN_DATA_DIR, 'clean_submission_data.csv')
    clean_training_data = pd.read_csv(clean_training_data_path)
    clean_submission_data = pd.read_csv(clean_submission_data_path)

    # Split data
    X = clean_training_data.drop(columns=target_variable).values
    y = clean_training_data[target_variable].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SPLIT_SEED)

    # Remove outliers from training set
    X_train, y_train = remove_outliers(X_train, y_train)

    ''' 2. Create and train model(s) '''
    models = [
        ('GradientBoostingRegressor', GradientBoostingRegressor(
            n_estimators=400, max_depth=4, max_features=0.9, warm_start=True)),
        ('XGBRegressor', XGBRegressor(
            objective='reg:squarederror', n_estimators=400, max_depth=4, subsample=0.9)),
        ('Deep Neural Network',
            create_nn_model(X_train.shape[1]))
    ]

for name, model in models:
    # Train model on training dataset
    start_time = time.time()
    print('--- Starting Training ({}) ---'.format(name))
    train_model(model, X_train, y_train)
    training_time = time.time() - start_time
    print('--- Training took {} sec ---'.format(training_time))

    # Test model
    print('--- Testing ({}) ---'.format(name))
    train_accuracy, train_pred = test_model(model, X_train, y_train)
    test_accuracy, test_pred = test_model(model, X_test, y_test)
    print('RMSE Train: {}, RMSE Test: {}'.format(
        train_accuracy, test_accuracy))

    # Train model on full dataset
    start_time = time.time()
    print('--- Starting Training (Full dataset) ---')
    train_model(model, X, y)
    print('--- Training took {} sec ---'.format(time.time() - start_time))

    # Test model after full dataset training
    print('--- Testing ({}) (Full dataset) ---'.format(name))
    full_accuracy, full_pred = test_model(model, X, y)
    print('RMSE Full: {}'.format(full_accuracy))

    # Save model
    archive_model(name, model, predictions=[train_pred, test_pred, full_pred],
                  performances=[train_accuracy,
                                test_accuracy, full_accuracy],
                  training_time=training_time, submission_data=clean_submission_data)

    print('\n')
