import itertools
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def k_nearest(X_train, y_train, X_validate, y_validate):
    
    # define variables
    metrics = []
    for k in range(1,21):
        # make
        knn = KNeighborsClassifier(n_neighbors=k)
        # fit
        knn.fit(X_train, y_train)
        # use
        train_score = knn.score(X_train, y_train)
        validate_score = knn.score(X_validate, y_validate)

        diff_score = train_score - validate_score

        output = pd.DataFrame({
            'k': [k],
            'train_score': [train_score],
            'validate_score': [validate_score],
            'diff_score': [diff_score]
        })

        metrics.append(output)

    baseline_accuracy = (y_train == 0).mean()

    results = pd.concat(metrics, ignore_index=True)

    sns.set_style('whitegrid')
    results.set_index('k').plot(figsize=(16,9))
    plt.ylabel('Accuracy')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xticks(np.arange(0,21,1))
    
    plt.legend()
    plt.show()
    
    return results


def k_nearest1(X_train, y_train, X_validate, y_validate):
    
    # define variables
    metrics = []
    for k in range(1,21):
        # make
        knn = KNeighborsClassifier(n_neighbors=k)
        # fit
        knn.fit(X_train, y_train)
        # use
        train_score = knn.score(X_train, y_train)
        validate_score = knn.score(X_validate, y_validate)

        diff_score = train_score - validate_score

        output = pd.DataFrame({
            'k': [k],
            'train_score': [train_score],
            'validate_score': [validate_score],
            'diff_score': [diff_score]
        })

        metrics.append(output)

    baseline_accuracy = (y_train == 0).mean()

    results = pd.concat(metrics, ignore_index=True)

    sns.set_style('whitegrid')
    ax = results.set_index('k').plot(figsize=(16,9))
    plt.ylabel('Accuracy')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xticks(np.arange(0,21,1))
    
    min_diff_idx = np.abs(results['diff_score']).argmin()
    min_diff_k = results.loc[min_diff_idx, 'k']
    min_diff_score = results.loc[min_diff_idx, 'diff_score']
    
    ax.axvline(min_diff_k, linestyle='--', linewidth=2, color='red', label=f'min diff at k={min_diff_k} (diff={min_diff_score:.3f})')
    
    plt.legend()
    plt.show()
    
    return results


def k_nearest2(X_train, y_train, X_validate, y_validate):
    """
    Trains and evaluates KNN models for different values of k and plots the results.

    Parameters:
    -----------
    X_train: array-like, shape (n_samples, n_features)
        Training input samples.
    y_train: array-like, shape (n_samples,)
        Target values for the training input samples.
    X_validate: array-like, shape (n_samples, n_features)
        Validation input samples.
    y_validate: array-like, shape (n_samples,)
        Target values for the validation input samples.

    Returns:
    --------
    results: pandas DataFrame
        Contains the train and validation accuracy for each value of k.
    """
    metrics = []
    train_score = []
    validate_score = []
    for k in range(1,21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_score.append(knn.score(X_train, y_train))
        validate_score.append(knn.score(X_validate, y_validate))
        diff_score = train_score[-1] - validate_score[-1]
        metrics.append({'k': k, 'train_score': train_score[-1], 'validate_score': validate_score[-1], 'diff_score': diff_score})

    baseline_accuracy = (y_train == 0).mean()

    results = pd.DataFrame.from_records(metrics)

    with sns.axes_style('whitegrid'):
        ax = results.set_index('k').plot(figsize=(16,9))
        plt.ylabel('Accuracy')
        plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
        plt.xticks(np.arange(0,21,1))
        min_diff_idx = np.abs(results['diff_score']).argmin()
        min_diff_k = results.loc[min_diff_idx, 'k']
        min_diff_score = results.loc[min_diff_idx, 'diff_score']
        ax.axvline(min_diff_k, linestyle='--', linewidth=2, color='red', label=f'min diff at k={min_diff_k} (diff={min_diff_score:.3f})')
        plt.legend()
        plt.show()
    
    return results


    
    
