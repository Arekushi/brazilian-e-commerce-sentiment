import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import re
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_val_predict, learning_curve
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import time
from datetime import datetime

nltk.download('stopwords')
nltk.download('rslp')


def import_data(path, sep=',', optimized=True, n_lines=50, encoding='utf-8', usecols=None, verbose=True):
    if optimized:
        df_raw = pd.read_csv(path, sep=sep, nrows=n_lines, encoding=encoding, usecols=usecols)
        start_mem = df_raw.memory_usage().sum() / 1024 ** 2

        # Columns were the optimization is applicable
        float64_cols = [col for col, dtype in df_raw.dtypes.items() if dtype == 'float64']
        int64_cols = [col for col, dtype in df_raw.dtypes.items() if dtype == 'int64']
        total_opt = len(float64_cols) + len(int64_cols)
        if verbose:
            print(f'This dataset has {df_raw.shape[1]} columns, which {total_opt} is/are applicable to optimization.\n')

        # Optimizing data types: float64 to float32
        for col in float64_cols:
            df_raw[col] = df_raw[col].astype('float32')

        for col in int64_cols:
            df_raw[col] = df_raw[col].astype('int32')

        if verbose:
            print('----------------------------------------------------')
            print(f'Memory usage ({n_lines} lines): {start_mem:.4f} MB')
            end_mem = df_raw.memory_usage().sum() / 1024 ** 2
            print(f'Memory usage after optimization ({n_lines} lines): {end_mem:.4f} MB')
            print('----------------------------------------------------')
            mem_reduction = 100 * (1 - (end_mem / start_mem))
            print(f'\nReduction of {mem_reduction:.2f}% on memory usage\n')

        dtypes = df_raw.dtypes
        col_names = dtypes.index
        types = [dtype.name for dtype in dtypes.values]
        column_types = dict(zip(col_names, types))

        try:
            return pd.read_csv(path, sep=sep, dtype=column_types, encoding=encoding, usecols=usecols)
        except ValueError as e1:
            print(f'ValueError on data reading: {e1}')
            print('The dataset will be read without optimization types.')
            return pd.read_csv(path, sep=sep, encoding=encoding, usecols=usecols)
    else:
        return pd.read_csv(path, sep=sep, encoding=encoding, usecols=usecols)



class ColumnMapping(BaseEstimator, TransformerMixin):
    def __init__(self, old_col_name, mapping_dict, new_col_name='target', drop=True):
        self.old_col_name = old_col_name
        self.mapping_dict = mapping_dict
        self.new_col_name = new_col_name
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.new_col_name] = X[self.old_col_name].map(self.mapping_dict)

        if self.drop:
            X.drop(self.old_col_name, axis=1, inplace=True)

        return X


class DropNullData(BaseEstimator, TransformerMixin):
    def __init__(self, cols_dropna=None):
        self.cols_dropna = cols_dropna

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.cols_dropna is not None:
            X[self.cols_dropna] = X[self.cols_dropna].dropna()
            return X
        else:
            return X.dropna()


class DropDuplicates(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df.drop_duplicates()


def re_breakline(text_list, text_sub=' '):
    return [re.sub('[\n\r]', text_sub, r) for r in text_list]


def re_hiperlinks(text_list, text_sub=' link '):
    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return [re.sub(pattern, text_sub, r) for r in text_list]


def re_dates(text_list, text_sub=' data '):
    pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    return [re.sub(pattern, text_sub, r) for r in text_list]

def re_money(text_list, text_sub=' dinheiro '):
    pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
    return [re.sub(pattern, text_sub, r) for r in text_list]


def re_numbers(text_list, text_sub=' numero '):
    return [re.sub('[0-9]+', text_sub, r) for r in text_list]


def re_negation(text_list, text_sub=' nega????o '):
    return [re.sub('([nN][????aA][oO]|[????]| [nN] )', text_sub, r) for r in text_list]


def re_special_chars(text_list, text_sub=' '):
    return [re.sub('\W', text_sub, r) for r in text_list]


def re_whitespaces(text_list):
    white_spaces = [re.sub('\s+', ' ', r) for r in text_list]
    white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
    return white_spaces_end


def stopwords_removal(text, cached_stopwords=stopwords.words('portuguese')):
    return [c.lower() for c in text.split() if c.lower() not in cached_stopwords]


def stemming_process(text, stemmer=RSLPStemmer()):
    return [stemmer.stem(c) for c in text.split()]


class ApplyRegex(BaseEstimator, TransformerMixin):

    def __init__(self, regex_transformers):
        self.regex_transformers = regex_transformers

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for regex_name, regex_function in self.regex_transformers.items():
            X = regex_function(X)

        return X


class StopWordsRemoval(BaseEstimator, TransformerMixin):

    def __init__(self, text_stopwords):
        self.text_stopwords = text_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [' '.join(stopwords_removal(comment, self.text_stopwords)) for comment in X]


class StemmingProcess(BaseEstimator, TransformerMixin):

    def __init__(self, stemmer):
        self.stemmer = stemmer

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [' '.join(stemming_process(comment, self.stemmer)) for comment in X]



class BinaryClassification:
    def __init__(self):
        self.classifiers_info = {}

    def fit(self, classifiers, X, y, approach='', random_search=False, scoring='roc_auc', cv=5, verbose=5, n_jobs=-1):
        for model_name, model_info in classifiers.items():
            clf_key = model_name + approach
            self.classifiers_info[clf_key] = {}

            if random_search:
                rnd_search = RandomizedSearchCV(model_info['model'], model_info['params'], scoring=scoring, cv=cv,
                                                verbose=verbose, random_state=42, n_jobs=n_jobs)
                rnd_search.fit(X, y)

                # Saving the best estimator into the model's dict
                self.classifiers_info[clf_key]['estimator'] = rnd_search.best_estimator_
            else:
                self.classifiers_info[clf_key]['estimator'] = model_info['model'].fit(X, y)

    def compute_train_performance(self, model_name, estimator, X, y, cv=5):
        t0 = time.time()
        accuracy = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy').mean()
        precision = cross_val_score(estimator, X, y, cv=cv, scoring='precision').mean()
        recall = cross_val_score(estimator, X, y, cv=cv, scoring='recall').mean()
        f1 = cross_val_score(estimator, X, y, cv=cv, scoring='f1').mean()

        # Probas for calculating AUC
        try:
            y_scores = cross_val_predict(estimator, X, y, cv=cv, method='decision_function')
        except:
            # Tree based models don't have 'decision_function()' method, but 'predict_proba()'
            y_probas = cross_val_predict(estimator, X, y, cv=cv, method='predict_proba')
            y_scores = y_probas[:, 1]
        auc = roc_auc_score(y, y_scores)

        # Saving scores on self.classifiers_info dictionary
        self.classifiers_info[model_name]['train_scores'] = y_scores

        t1 = time.time()
        delta_time = t1 - t0
        train_performance = {}
        train_performance['model'] = model_name
        train_performance['approach'] = f'Treino {cv} K-folds'
        train_performance['acc'] = round(accuracy, 4)
        train_performance['precision'] = round(precision, 4)
        train_performance['recall'] = round(recall, 4)
        train_performance['f1'] = round(f1, 4)
        train_performance['auc'] = round(auc, 4)
        train_performance['total_time'] = round(delta_time, 3)

        return pd.DataFrame(train_performance, index=train_performance.keys()).reset_index(drop=True).loc[:0, :]

    def compute_test_performance(self, model_name, estimator, X, y):
        t0 = time.time()
        y_pred = estimator.predict(X)
        y_proba = estimator.predict_proba(X)
        y_scores = y_proba[:, 1]

        # Retrieving metrics using test data
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_scores)

        # Saving probabilities on treined classifiers dictionary
        self.classifiers_info[model_name]['test_scores'] = y_scores

        # Creating a DataFrame with metrics
        t1 = time.time()
        delta_time = t1 - t0
        test_performance = {}
        test_performance['model'] = model_name
        test_performance['approach'] = f'Teste'
        test_performance['acc'] = round(accuracy, 4)
        test_performance['precision'] = round(precision, 4)
        test_performance['recall'] = round(recall, 4)
        test_performance['f1'] = round(f1, 4)
        test_performance['auc'] = round(auc, 4)
        test_performance['total_time'] = round(delta_time, 3)

        return pd.DataFrame(test_performance, index=test_performance.keys()).reset_index(drop=True).loc[:0, :]

    def evaluate_performance(self, X_train, y_train, X_test, y_test, cv=5, save=False, overwrite=True,
                             performances_filepath='model_performances.csv'):
        df_performances = pd.DataFrame({})
        for model_name, model_info in self.classifiers_info.items():

            # Validating if the model was already trained (the key 'train_performance' will be at model_info dict if so)
            if 'train_performance' in model_info.keys():
                df_performances = df_performances.append(model_info['train_performance'])
                df_performances = df_performances.append(model_info['test_performance'])
                continue

            # Returning the estimator for calling the evaluation functions
            #print(f'Evaluating model {model_name}\n')
            estimator = model_info['estimator']

            # Retrieving training and testing metrics by calling inner functions
            train_performance = self.compute_train_performance(model_name, estimator, X_train, y_train, cv=cv)
            test_performance = self.compute_test_performance(model_name, estimator, X_test, y_test)

            # Putting results on model's dictionary (classifiers_info)
            self.classifiers_info[model_name]['train_performance'] = train_performance
            self.classifiers_info[model_name]['test_performance'] = test_performance

            # Building and unique DataFrame with performances retrieved
            model_performance = train_performance.append(test_performance)
            df_performances = df_performances.append(model_performance)

            # Saving some attributes on model_info dictionary for further access
            model_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
            model_info['model_data'] = model_data

        # Saving the metrics file if applicable
        if save:
            # Adding information of measuring and execution time
            cols_performance = list(df_performances.columns)
            df_performances['anomesdia'] = datetime.now().strftime('%Y%m%d')
            df_performances['anomesdia_datetime'] = datetime.now()
            df_performances = df_performances.loc[:, ['anomesdia', 'anomesdia_datetime'] + cols_performance]

            # Validating overwriting or append on data already saved
            if overwrite:
                df_performances.to_csv(performances_filepath, index=False)
            else:
                try:
                    # If overwrite is False, tries reading existing metrics data and applying append on it
                    log_performances = pd.read_csv(performances_filepath)
                    full_performances = log_performances.append(df_performances)
                    full_performances.to_csv(performances_filepath, index=False)
                except FileNotFoundError:
                    print('Log de performances do modelo n??o existente no caminho especificado. Salvando apenas o atual.')
                    df_performances.to_csv(performances_filepath, index=False)

        return df_performances

    def feature_importance_analysis(self, features, specific_model=None, graph=True, ax=None, top_n=30,
                                    palette='viridis', save=False, features_filepath='features_info.csv'):
        """
        This function retrieves the feature importance from a given model. It can also build a bar chart
        for top_n most important features and plot it on the notebook.

        Paramters
        ---------
        :param features: list of model features used on training [type: list]
        :param specific_model: information that guides the returning of feature importance for a specific model*
        :param graph: flag that guides bar chart plotting at the end of execution [type: bool, default: True]
        :param ax: axis for plotting the bar chart [type: matplotlib.axes, default: None]
        :param top_n: parameter for showing up just top most important features [type: int, default: 30]
        :param palette: color configuration for feature importance bar chart [type: string, default: 'viridis']
        :param save: flag for saving the dataset returned [type: bool, default: False]
        :param features_filepath: path for saving the feature iportance dataset [type: string, default: 'features_info.csv']

        Returns
        -------
        :return: model_feature_importance: pandas DataFrame with feature importances extracted by trained models
        """

        # Iterating over each trained classifiers on classifiers_info dictionary
        feat_imp = pd.DataFrame({})
        all_feat_imp = pd.DataFrame({})
        for model_name, model_info in self.classifiers_info.items():
            # Creating a pandas DataFrame with model feature importance
            try:
                importances = model_info['estimator'].feature_importances_
            except:
                # If the given model doesn't have the feature_importances_ method, just continue for the next
                continue
            # Preparing the dataset with useful information
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp['anomesdia'] = datetime.now().strftime('%Y%m')
            feat_imp['anomesdia_datetime'] = datetime.now()
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)
            feat_imp.reset_index(drop=True, inplace=True)

            # Saving the feature iportance at model's dictionary (classifiers_info)
            self.classifiers_info[model_name]['feature_importances'] = feat_imp
            all_feat_imp = all_feat_imp.append(feat_imp)
            all_feat_imp['model'] = model_name

        # Retrieving feature importance for a specific model
        if specific_model is not None:
            try:
                model_feature_importance = self.classifiers_info[specific_model]['feature_importances']
                if graph:
                    # Plotting the bar chart
                    sns.barplot(x='importance', y='feature', data=model_feature_importance.iloc[:top_n, :],
                                ax=ax, palette=palette)
                    format_spines(ax, right_border=False)
                    ax.set_title(f'Top {top_n} {specific_model} Features mais Relevantes', size=14, color='dimgrey')

                # Saving features for a specific model
                if save:
                    model_feature_importance['model'] = specific_model
                    order_cols = ['anomesdia', 'anomesdia_datetime', 'model', 'feature', 'importance']
                    model_feature_importance = model_feature_importance.loc[:, order_cols]
                    model_feature_importance.to_csv(features_filepath, index=False)
                return model_feature_importance

            except:
                # Exception raised if the "specific_model" param doesn't match with any model's dictionary key
                print(f'Classificador {specific_model} n??o existente nas chaves de classificadores treinados.')
                print(f'Op????es poss??veis: {list(self.classifiers_info.keys())}')
                return None

        else:
            # Validating feature importance saving if not passing specific_model param
            if save:
                order_cols = ['anomesdia', 'anomedia_datetime', 'model', 'feature', 'importance']
                all_feat_imp = all_feat_imp.loc[:, order_cols]
                all_feat_imp.to_csv(features_filepath, index=False)
            return all_feat_imp

        # Non-matching param combination (it can't be possible plotting bar chart for all models)
        if graph and specific_model is None:
            print('Por favor, escolha um modelo espec??fico para visualizar o gr??fico das feature importances')
            return None

    def plot_roc_curve(self, figsize=(16, 6)):
        """
        This function iterates over each estimator in classifiers_info dictionary and plots the ROC Curve for
        each one for training (first axis) and testing data (second axis)

        Paramaters
        ----------
        :param figsize: figure size for the plot [type: tuple, default: (16, 6)]

        Returns
        -------
        This function doesn't return anything but the matplotlib plot for ROC Curve

        Application
        -----------
        trainer.plot_roc_curve()
        """

        # Creating matplotlib figure and axis for ROC Curve plot
        fig, axs = plt.subplots(ncols=2, figsize=figsize)

        # Iterating over trained models
        for model_name, model_info in self.classifiers_info.items():
            # Returning y data for the model (training and testing)
            y_train = model_info['model_data']['y_train']
            y_test = model_info['model_data']['y_test']

            # Returning scores already calculated after performance evaluation
            train_scores = model_info['train_scores']
            test_scores = model_info['test_scores']

            # Calculating false positives and true positives rate
            train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_scores)
            test_fpr, test_tpr, test_thresholds = roc_curve(y_test, test_scores)

            # Returning the auc metric for training and testing already calculated after model evaluation
            train_auc = model_info['train_performance']['auc'].values[0]
            test_auc = model_info['test_performance']['auc'].values[0]

            # Plotting graph (training data)
            plt.subplot(1, 2, 1)
            plt.plot(train_fpr, train_tpr, linewidth=2, label=f'{model_name} auc={train_auc}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis([-0.02, 1.02, -0.02, 1.02])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Train Data')
            plt.legend()

            # Plotting graph (testing data)
            plt.subplot(1, 2, 2)
            plt.plot(test_fpr, test_tpr, linewidth=2, label=f'{model_name} auc={test_auc}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis([-0.02, 1.02, -0.02, 1.02])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Test Data', size=12)
            plt.legend()

        plt.show()

    def custom_confusion_matrix(self, model_name, y_true, y_pred, classes, cmap, normalize=False):
        """
        This function is used for plotting and customizing a confusion matrix for a specific model. In practice,
        this function can be called by a top level one for plotting matrix for many models.

        Parameters
        ----------
        :param model_name: key reference for extracting model's estimator from classifiers_dict [type: string]
        :param y_true: label reference for the target variable [type: np.array]
        :param y_pred: array of predictions given by the respective model [type: np.array]
        :param classes: alias for classes [type: string]
        :param cmap: this parameters guides the colorway for the matrix [type: matplotlib.colormap]
        :param normalize: normalizes the entries for the matrix [type: bool, default: False]

        Returns
        -------
        :return: This functions doesn't return any object besides of plotting the confusion matrix

        Application
        -----------
        Please refer to the self.plot_confusion_matrix() function

        """

        # Returning a confusion matrix given the labels and predictions passed as args
        conf_mx = confusion_matrix(y_true, y_pred)

        # Plotando matriz
        plt.imshow(conf_mx, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))

        # Customizando eixos
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Customizando entradas
        fmt = '.2f' if normalize else 'd'
        thresh = conf_mx.max() / 2.
        for i, j in itertools.product(range(conf_mx.shape[0]), range(conf_mx.shape[1])):
            plt.text(j, i, format(conf_mx[i, j]),
                     horizontalalignment='center',
                     color='white' if conf_mx[i, j] > thresh else 'black')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'{model_name}\nConfusion Matrix', size=12)

    def plot_confusion_matrix(self, classes, normalize=False, cmap=plt.cm.Blues):
        """
        This function plots a confusion matrix for training and testing data for each classifier at
        self.classifiers_dict dictionary

        Parameters
        ----------
        :param classes: labels for the target variable [type: string]
        :param normalize: flag that guides the normalization of matrix values [type: bool, default: False]
        :param cmap: param that colorizes the matrix [type: plt.cm, default: plt.cm.Blues]

        Returns
        -------
        This function doesn't return anything but the matplotlib plot for confusion matrix
        """

        # Defining parameters for ploting
        k = 1
        nrows = len(self.classifiers_info.keys())
        fig = plt.figure(figsize=(10, nrows * 4))
        sns.set(style='white', palette='muted', color_codes=True)

        # Iterating over each classifier
        for model_name, model_info in self.classifiers_info.items():
            # Returning data from each model
            X_train = model_info['model_data']['X_train']
            y_train = model_info['model_data']['y_train']
            X_test = model_info['model_data']['X_test']
            y_test = model_info['model_data']['y_test']

            # Making predictions for training (cross validation) and testing for returning confusion matrix
            train_pred = cross_val_predict(model_info['estimator'], X_train, y_train, cv=5)
            test_pred = model_info['estimator'].predict(X_test)

            # Plotting matrix (training data)
            plt.subplot(nrows, 2, k)
            self.custom_confusion_matrix(model_name + ' Train', y_train, train_pred, classes=classes, cmap=cmap,
                                         normalize=normalize)
            k += 1

            # Plotting matrix (testing data)
            plt.subplot(nrows, 2, k)
            self.custom_confusion_matrix(model_name + ' Test', y_test, test_pred, classes=classes, cmap=plt.cm.Greens,
                                         normalize=normalize)
            k += 1

        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self, model_name, ax, ylim=None, cv=5, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
        """
        This function calculates and plots the learning curve for a trained model

        Parameters
        ----------
        :param model_name: Key reference for extracting an estimator from classifiers_dict dictionary [type: string]
        :param ax: axis reference for plotting the learning curve [type: matplotlib.axis]
        :param ylim: configuration of the limit on plot vertical axis [type: int, default: None]
        :param cv: k-folds used on cross validation [type: int, default: 5]
        :param n_jobs: number of cores used on retrieving the learning curve params [type: int, default: 1]
        :param train_sizes: array that guides the steps bins used on learning curve [type: np.array,
                                                                                    default:np.linspace(.1, 1.0, 10)]

        Returns
        -------
        This function doesn't return anything but the matplotlib plot for the learning curve

        Application
        -----------
        # Plotting the learning curve for a specific model
        fig, ax = plt.subplots(figsize=(16, 6))
        trainer.plot_learning_curve(model_name='LightGBM', ax=ax)
        """

        # Returning the model to be evaluated
        try:
            model = self.classifiers_info[model_name]
        except:
            print(f'Classificador {model_name} n??o foi treinado.')
            print(f'Op????es poss??veis: {list(self.classifiers_info.keys())}')
            return None

        # Returning useful data for the model
        X_train = model['model_data']['X_train']
        y_train = model['model_data']['y_train']

        # Calling the learning curve model for retrieving the scores for training and validation
        train_sizes, train_scores, val_scores = learning_curve(model['estimator'], X_train, y_train, cv=cv,
                                                               n_jobs=n_jobs, train_sizes=train_sizes)

        # Computing averages and standard deviation (training and validation)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        # Results on training data
        ax.plot(train_sizes, train_scores_mean, 'o-', color='navy', label='Training Score')
        ax.fill_between(train_sizes, (train_scores_mean - train_scores_std), (train_scores_mean + train_scores_std),
                        alpha=0.1, color='blue')

        # Results on cross validation
        ax.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Cross Val Score')
        ax.fill_between(train_sizes, (val_scores_mean - val_scores_std), (val_scores_mean + val_scores_std),
                        alpha=0.1, color='crimson')

        # Customizing graph
        ax.set_title(f'Model {model_name} - Learning Curve', size=14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc='best')

    def plot_score_distribution(self, model_name, shade=False):
        """
        This function plots a kdeplot for training and testing data splitting by target class

        Parameters
        ----------
        :param model_name: key reference for the trained model [type: string]
        :param shade: shade param for seaborn's kdeplot [type: bool, default: False]

        Returns
        -------
        This function doesn't return anything but the matplotlib plot for the score distribution

        Application
        -----------
        # Ploting scores distribution for a model
        plot_score_distribution(model_name='LightGBM', shade=True)
        """

        # Returning the model to be evaluated
        try:
            model = self.classifiers_info[model_name]
        except:
            print(f'Classificador {model_name} n??o foi treinado.')
            print(f'Op????es poss??veis: {list(self.classifiers_info.keys())}')
            return None

        # Retrieving y array for training and testing data
        y_train = model['model_data']['y_train']
        y_test = model['model_data']['y_test']

        # Retrieving training and testing scores
        train_scores = model['train_scores']
        test_scores = model['test_scores']

        # Plotting scores distribution
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
        sns.kdeplot(train_scores[y_train == 1], ax=axs[0], label='y=1', shade=shade, color='darkslateblue')
        sns.kdeplot(train_scores[y_train == 0], ax=axs[0], label='y=0', shade=shade, color='crimson')
        sns.kdeplot(test_scores[y_test == 1], ax=axs[1], label='y=1', shade=shade, color='darkslateblue')
        sns.kdeplot(test_scores[y_test == 0], ax=axs[1], label='y=0', shade=shade, color='crimson')

        # Customizing plots
        format_spines(axs[0], right_border=False)
        format_spines(axs[1], right_border=False)
        axs[0].set_title('Score Distribution - Training Data', size=12, color='dimgrey')
        axs[1].set_title('Score Distribution - Testing Data', size=12, color='dimgrey')
        plt.suptitle(f'Score Distribution: a Probability Approach for {model_name}\n', size=14, color='black')
        plt.show()

    def plot_score_bins(self, model_name, bin_range):
        """
        This function plots a score distribution based on quantity of each class in a specific bin_range set

        Parameters
        ----------
        :param model_name: key reference for the trained model [type: string]
        :param bin_range: defines a range of splitting the bins array [type: float]

        Returns
        -------
        This function doesn't return anything but the matplotlib plot for the score bins distribution

        Application
        -----------
        # Ploting scores distribution for a model in another approach
        plot_score_bins(model_name='LightGBM', bin_range=0.1)
        """

        # Returning the model to be evaluated
        try:
            model = self.classifiers_info[model_name]
        except:
            print(f'Classificador {model_name} n??o foi treinado.')
            print(f'Op????es poss??veis: {list(self.classifiers_info.keys())}')
            return None

        # Creating the bins array
        bins = np.arange(0, 1.01, bin_range)
        bins_labels = [str(round(list(bins)[i - 1], 2)) + ' a ' + str(round(list(bins)[i], 2)) for i in range(len(bins))
                       if i > 0]

        # Retrieving the train scores and creating a DataFrame
        train_scores = model['train_scores']
        y_train = model['model_data']['y_train']
        df_train_scores = pd.DataFrame({})
        df_train_scores['scores'] = train_scores
        df_train_scores['target'] = y_train
        df_train_scores['faixa'] = pd.cut(train_scores, bins, labels=bins_labels)

        # Computing the distribution for each bin
        df_train_rate = pd.crosstab(df_train_scores['faixa'], df_train_scores['target'])
        df_train_percent = df_train_rate.div(df_train_rate.sum(1).astype(float), axis=0)

        # Retrieving the test scores and creating a DataFrame
        test_scores = model['test_scores']
        y_test = model['y_test']
        df_test_scores = pd.DataFrame({})
        df_test_scores['scores'] = test_scores
        df_test_scores['target'] = y_test
        df_test_scores['faixa'] = pd.cut(test_scores, bins, labels=bins_labels)

        # Computing the distribution for each bin
        df_test_rate = pd.crosstab(df_test_scores['faixa'], df_test_scores['target'])
        df_test_percent = df_test_rate.div(df_test_rate.sum(1).astype(float), axis=0)

        # Defining figure for plotting
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

        # Plotting the bar chart for each bin
        for df_scores, ax in zip([df_train_scores, df_test_scores], [axs[0, 0], axs[0, 1]]):
            sns.countplot(x='faixa', data=df_scores, hue='target', ax=ax, palette=['darkslateblue', 'crimson'])
            AnnotateBars(n_dec=0, color='dimgrey').vertical(ax)
            ax.legend(loc='upper right')
            format_spines(ax, right_border=False)

        # Plotting percentage for each class in each bin
        for df_percent, ax in zip([df_train_percent, df_test_percent], [axs[1, 0], axs[1, 1]]):
            df_percent.plot(kind='bar', ax=ax, stacked=True, color=['darkslateblue', 'crimson'], width=0.6)

            # Customizing plots
            for p in ax.patches:
                # Collecting params for labeling
                height = p.get_height()
                width = p.get_width()
                x = p.get_x()
                y = p.get_y()

                # Formatting params and putting into the graph
                label_text = f'{round(100 * height, 1)}%'
                label_x = x + width - 0.30
                label_y = y + height / 2
                ax.text(label_x, label_y, label_text, ha='center', va='center', color='white',
                        fontweight='bold', size=10)
            format_spines(ax, right_border=False)

        # Final definitions
        axs[0, 0].set_title('Quantity of each Class by Range - Train', size=12, color='dimgrey')
        axs[0, 1].set_title('Quantity of each Class by Range - Test', size=12, color='dimgrey')
        axs[1, 0].set_title('Percentage of each Class by Range - Train', size=12, color='dimgrey')
        axs[1, 1].set_title('Percentage of each Class by Range - Test', size=12, color='dimgrey')
        plt.suptitle(f'Score Distribution by Range - {model_name}\n', size=14, color='black')
        plt.tight_layout()
        plt.show()

    def shap_analysis(self, model_name, features):
        """
        This function brings a shap analysis for each feature into the model

        Parameters
        ----------
        :param model_name: key reference for the trained model [type: string]
        :param features: features list for the model [type: list]

        Returns
        -------
        This function doesn't return anything but the shap plot analysis

        Application
        -----------
        # Executing a shap analysis
        trainer.shap_analysis(model_name='LightGBM')
        """

        # Returning the model to be evaluated
        try:
            model = self.classifiers_info[model_name]
        except:
            print(f'Classificador {model_name} n??o foi treinado.')
            print(f'Op????es poss??veis: {list(self.classifiers_info.keys())}')
            return None

        # Retrieving training data
        X_train = model['X_train']

        # Applying shap approach
        explainer = shap.TreeExplainer(model)
        df_train = pd.DataFrame(X_train, columns=features)
        shap_values = explainer.shap_values(df_train)

        # Plotting a summary plot using shap
        shap.summary_plot(shap_values[1], df_train)


def cross_val_performance(estimator, X, y, cv=5):
    """
    This function applies cross validation to retrieve useful metrics for the classification model.
    In practice, this function would be called by another one (usually with compute_test_performance as well)

    Parameters
    ----------
    :param estimator: a trained model to be used on evaluation [type: model]
    :param X: object containing features already prepared for training the model [type: np.array]
    :param y: object containing the model target variable [type: np.array]
    :param cv: k-folds for cross validation application on training evaluation

    Return
    ------
    :return train_performance: DataFrame containing model metrics calculated using cross validation

    Application
    -----------
    # Evaluating training performance using cross-validation
    df_performances = trainer.compute_train_performance('DecisionTrees', trained_model, X_train, y_train, cv=5)
    """

    # Computing metrics using cross validation
    t0 = time.time()
    accuracy = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy').mean()
    precision = cross_val_score(estimator, X, y, cv=cv, scoring='precision').mean()
    recall = cross_val_score(estimator, X, y, cv=cv, scoring='recall').mean()
    f1 = cross_val_score(estimator, X, y, cv=cv, scoring='f1').mean()

    # Probas for calculating AUC
    try:
        y_scores = cross_val_predict(estimator, X, y, cv=cv, method='decision_function')
    except:
        # Tree based models don't have 'decision_function()' method, but 'predict_proba()'
        y_probas = cross_val_predict(estimator, X, y, cv=cv, method='predict_proba')
        y_scores = y_probas[:, 1]
    auc = roc_auc_score(y, y_scores)

    # Creating a DataFrame with metrics
    t1 = time.time()
    delta_time = t1 - t0
    train_performance = {}
    train_performance['model'] = estimator.__class__.__name__
    train_performance['approach'] = 'Final Model'
    train_performance['acc'] = round(accuracy, 4)
    train_performance['precision'] = round(precision, 4)
    train_performance['recall'] = round(recall, 4)
    train_performance['f1'] = round(f1, 4)
    train_performance['auc'] = round(auc, 4)
    train_performance['total_time'] = round(delta_time, 3)
    df_performance = pd.DataFrame(train_performance, index=train_performance.keys()).reset_index(drop=True).loc[:0, :]

    # Adding information of measuring and execution time
    cols_performance = list(df_performance.columns)
    df_performance['anomesdia'] = datetime.now().strftime('%Y%m%d')
    df_performance['anomesdia_datetime'] = datetime.now()
    df_performance = df_performance.loc[:, ['anomesdia', 'anomesdia_datetime'] + cols_performance]

    return df_performance
