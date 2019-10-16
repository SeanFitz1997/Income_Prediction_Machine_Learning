import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import CLEAN_DATA_DIR

DIAGRAM_OUTPUT_MESSAGE = '{} saved to {}'


def view_data_overview(df):
    save_filepath = os.path.abspath('images/data.png')

    plt.figure()
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(df)
    plt.savefig(save_filepath)

    print(DIAGRAM_OUTPUT_MESSAGE.format('Data overview', save_filepath))


def view_correlation_matrix(df):
    columns = list(df.columns)
    save_filepath = os.path.abspath('images/correlation_matrix.png')

    plt.figure()
    correlation_matrix = np.corrcoef(df[columns].values.T)
    sns.set(font_scale=1.)
    sns.heatmap(correlation_matrix, cbar=True, annot=True,
                square=True, fmt='.2f', yticklabels=columns, xticklabels=columns)
    plt.savefig(save_filepath)

    print(DIAGRAM_OUTPUT_MESSAGE.format('correlation matrix', save_filepath))


def view_feature_distribution(feature):
    save_filepath = os.path.abspath(
        'images/{}_distribution.png'.format(feature.name))
    title = '{} Distribution'.format(feature.name)

    plt.figure()
    sns.distplot(feature)
    plt.title(title)
    plt.savefig(save_filepath)

    print(DIAGRAM_OUTPUT_MESSAGE.format(title, save_filepath))


def view_feature_boxplot(feature):
    save_filepath = os.path.abspath(
        'images/{}_boxplot.png'.format(feature.name))
    title = '{} Boxplot'.format(feature.name)

    plt.figure()
    sns.boxplot(feature)
    plt.title(title)
    plt.savefig(save_filepath)

    print(DIAGRAM_OUTPUT_MESSAGE.format(title, save_filepath))


if __name__ == '__main__':
    print('--- Creating Clean Data ---')

    # Get clean data
    clean_training_data_path = os.path.join(
        CLEAN_DATA_DIR, 'clean_training_data.csv')
    df = pd.read_csv(clean_training_data_path)

    # Overview of all features
    view_data_overview(df)
    view_correlation_matrix(df)

    # Create distributeion and box plots
    for col in ['Income in EUR', 'Body Height [cm]', 'Age', 'Size of City']:
        view_feature_distribution(df[col])
        view_feature_boxplot(df[col])
