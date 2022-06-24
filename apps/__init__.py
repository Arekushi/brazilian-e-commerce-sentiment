from flask import Flask
from pandas import DataFrame
from datetime import datetime

from apps.resources import associate_resources
from config import config


def create_app(config_name):
    app = Flask('api-user')
    app.config.from_object(config[config_name])
    associate_resources(app)

    return app
