from datetime import datetime
from pandas import DataFrame
from joblib import load
from os import getenv
from os.path import isfile
from dotenv import load_dotenv
from pathlib import Path


_ENV_FILE = '.env'
if isfile(_ENV_FILE):
    load_dotenv(_ENV_FILE)


class Sentimentor:

    def __init__(self, input_data):
        self.input_data = input_data
        self.pipeline = load(f'{Path().absolute()}/ml/pipelines/text_prep_pipeline.pkl')
        self.model = load(f'{Path().absolute()}/ml/models/sentiment_clf_model.pkl')

    def prep_input(self):
        if type(self.input_data) is str:
            self.input_data = [self.input_data]
        elif type(self.input_data) is DataFrame:
            self.input_data = list(self.input_data.iloc[:, 0].values)

        return self.pipeline.transform(self.input_data)

    def make_predictions(self):
        text_list = self.prep_input()
        pred = self.model.predict(text_list)
        proba = self.model.predict_proba(text_list)[:, 1]

        class_sentiment = ['Positive' if c == 1 else 'Negative' for c in pred]
        class_proba = [p if c == 1 else 1 - p for c, p in zip(pred, proba)]

        results = {
            'datetime_prediction': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'text_input': self.input_data,
            'prediction': pred,
            'class_sentiment': class_sentiment,
            'class_probability': class_proba
        }

        return results
