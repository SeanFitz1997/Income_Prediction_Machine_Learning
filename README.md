# Income Prediction Machine Learning
This is my code for a [kaggle competition](https://www.kaggle.com/c/tcdml1920-income-ind/) to predict a persons income in Euro base on their:
* Year of Record
* Gender
* Age
* Country
* Size of City
* Profession 
* University Degree
* Wears Glasses
* Hair Color
* Body Height [cm]
* Income in EUR

## Whats Here
    |- data.py (Used to create clean data files)
    |- visualize.py (Used to create data visualizations)
    |- train.py (Used to train models and archive models and create submissions)
    |- data ¬
    |        |- Input data files ...
    |        |- clean_data (Clean training & submission data)
    |- images (data visualizations)
    |- models (trained models)

# Data preprocessing & Feature extraction
To create clean data csv files
```
python data.py
```
To clean the data I:
* Removed Instance, Hair Colour and Wears glasses features.
* Replaced all `NA` values with something meaningful (Mean for continuous values and move/ create new categories for class labels).
* One hot encode Gender and University values.
* Target encoded Country & Profession (i.e. Replaced country and profession values with their class mean income.). This target encoding can be smoothed with the mean target for cases with few class instances (See `get_target_mappings` in `data.py`). This allowed models to train a lot faster than with one hot encoding and significantly improved RMSE.

# Visualizations
To create data visualizations
```
python visualize.py
```
* Plots pairwise relationships of all features.
* Create correlation matrix of all features.
* Creates box plots for Income, Height, Age and Size of City.
* Creates distribution histograms for Income, Height, Age and Size of City.
See `images` directory

# Training
Train model(s)
```
python train.py
```
Trains models defined in models variable. Each model it trained, tested and saved. Performance, hyperparameter, training times, predictions and the serialized models are saved in the `models` directory.

To tain models I:
* Split the data into training & test sets.
* Train the model on the training set.
* Evaluate the models performance on the test set.
* Re-train model on the entire data set (To gain and extra performance from more data).

# Model performances
| Model                           | Train RMSE | Test RMSE  |
| ------------------------------- | -----------| ---------- |
| Deep Neural Network 128->128->1 | 83,330     | 89,5347    |
| Gradient Boosting Regression    | 47,765     | 56,427  	|
| XGB Regressor                   | 44,090     | 56,258     |
