import logging
from log.log_config import logger_config
from os import getenv
from os.path import isfile
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from ml.custom_transformers import *
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve


_ENV_FILE = '.env'
if isfile(_ENV_FILE):
    load_dotenv(dotenv_path=_ENV_FILE)


COLS_READ = ['review_comment_message', 'review_score']
CORPUS_COL = 'review_comment_message'
TARGET_COL = 'target'

PT_STOPWORDS = stopwords.words('portuguese')

WARNING_MESSAGE = f'Module {__file__} finished with ERROR status'


nltk.download('stopwords')
nltk.download('rslp')


logger = logging.getLogger(__name__)
logger = logger_config(logger, level=logging.DEBUG, filemode='a')


logger.debug('Reading raw data')
try:
    df = import_data(getenv('RAW_DATA'), usecols=COLS_READ, verbose=False)
except Exception as e:
    logger.error(e)
    logger.warning(WARNING_MESSAGE)
    exit()


score_map = {
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1
}

initial_prep_pipeline = Pipeline([
    ('mapper', ColumnMapping(old_col_name='review_score', mapping_dict=score_map, new_col_name=TARGET_COL)),
    ('null_dropper', DropNullData()),
    ('dup_dropper', DropDuplicates())
])

logger.debug('Applying initial_prep_pipeline on raw data')
try:
    df_prep = initial_prep_pipeline.fit_transform(df)
except Exception as e:
    logger.error(e)
    logger.warning(WARNING_MESSAGE)
    exit()


regex_transformers = {
    'break_line': re_breakline,
    'hiperlinks': re_hiperlinks,
    'dates': re_dates,
    'money': re_money,
    'numbers': re_numbers,
    'negation': re_negation,
    'special_chars': re_special_chars,
    'whitespaces': re_whitespaces
}

text_prep_pipeline = Pipeline([
    ('regex', ApplyRegex(regex_transformers)),
    ('stopwords', StopWordsRemoval(PT_STOPWORDS)),
    ('stemming', StemmingProcess(RSLPStemmer())),
    ('vectorizer', TfidfVectorizer(max_features=300, min_df=7, max_df=0.8, stop_words=PT_STOPWORDS))
])

logger.debug('Extracting X and y variables for training')
try:
    X = df_prep[CORPUS_COL].tolist()
    y = df_prep[TARGET_COL]
except Exception as e:
    logger.error(e)
    logger.warning(WARNING_MESSAGE)
    exit()

logger.debug('Applying text_prep_pipeline on X data')
try:
    X_prep = text_prep_pipeline.fit_transform(X)
except Exception as e:
    logger.error(e)
    logger.warning(WARNING_MESSAGE)
    exit()

X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=.20, random_state=42)

logreg_param_grid = {
    'C': np.linspace(0.1, 10, 20),
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced', None],
    'random_state': [42],
    'solver': ['liblinear']
}

set_classifiers = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': logreg_param_grid
    }
}

logger.debug('Training a sentiment classification model')
try:
    trainer = BinaryClassification()
    trainer.fit(set_classifiers, X_train, y_train, random_search=True, scoring='accuracy', verbose=0)
except Exception as e:
    logger.error(e)
    logger.warning(WARNING_MESSAGE)
    exit()


logger.debug('Evaluating models performance')
try:
    performance = trainer.evaluate_performance(X_train, y_train, X_test, y_test, cv=5, save=True, overwrite=True, 
                                                performances_filepath=getenv('MODEL_METRICS'))
except Exception as e:
    logger.error(e)
    logger.warning(WARNING_MESSAGE)
    exit()


model = trainer.classifiers_info[getenv('MODEL_KEY')]['estimator']

e2e_pipeline = Pipeline([
    ('text_prep', text_prep_pipeline),
    ('model', model)
])

param_grid = [{
    'text_prep__vectorizer__max_features': np.arange(600, 601, 50),
    'text_prep__vectorizer__min_df': [7],
    'text_prep__vectorizer__max_df': [.6]
}]

logger.debug('Searching for the best hyperparams combination')
try:
    grid_search_prep = GridSearchCV(e2e_pipeline, param_grid, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
    grid_search_prep.fit(X, y)
    logger.info(f'Done searching. The set of new hyperparams are: {grid_search_prep.best_params_}')
except Exception as e:
    logger.error(e)
    logger.warning(WARNING_MESSAGE)
    exit()

logger.debug('Updating model hyperparams')
try:
    vectorizer_max_features = grid_search_prep.best_params_['text_prep__vectorizer__max_features']
    vectorizer_min_df = grid_search_prep.best_params_['text_prep__vectorizer__min_df']
    vectorizer_max_df = grid_search_prep.best_params_['text_prep__vectorizer__max_df']

    e2e_pipeline.named_steps['text_prep'].named_steps['vectorizer'].max_features = vectorizer_max_features
    e2e_pipeline.named_steps['text_prep'].named_steps['vectorizer'].min_df = vectorizer_min_df
    e2e_pipeline.named_steps['text_prep'].named_steps['vectorizer'].max_df = vectorizer_max_df
except Exception as e:
    logger.error(e)
    logger.warning(WARNING_MESSAGE)
    exit()

logger.debug('Fitting the final model using the final pipeline')
try:
    e2e_pipeline.fit(X, y)
except Exception as e:
    logger.error(e)
    logger.warning(WARNING_MESSAGE)
    exit()


logger.debug('Evaluating final performance')
try:
    final_model = e2e_pipeline.named_steps['model']
    final_performance = cross_val_performance(final_model, X_prep, y, cv=5)
    final_performance = final_performance.append(performance)
    final_performance.to_csv(getenv('MODEL_METRICS'), index=False)
except Exception as e:
    logger.error(e)
    logger.warning(WARNING_MESSAGE)
    exit()


logger.debug('Saving pkl files')
try:
    dump(initial_prep_pipeline, getenv('INITIAL_PIPELINE'))
    dump(text_prep_pipeline, getenv('TEXT_PIPELINE'))
    dump(e2e_pipeline, getenv('E2E_PIPELINE'))
    dump(final_model, getenv('MODEL'))
    logger.info('Finished the module')
except Exception as e:
    logger.error(e)
    logger.warning(WARNING_MESSAGE)
    exit()
