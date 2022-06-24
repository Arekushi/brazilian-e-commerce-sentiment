from json import dumps
from flask import jsonify, request
from apps.sentimentor import Sentimentor


def associate_resources(app):
    @app.route('/')
    def index():
        return jsonify({
            'hello': 'world'
        })

    @app.route('/send')
    def home():
        input_data = request.json['message']
        sentimentor = Sentimentor(input_data=input_data)
        output = sentimentor.make_predictions()

        dict_return = {
            'input_data': input_data,
            'class_sentiment': output['class_sentiment'],
            'class_probability': output['class_probability']
        }

        return jsonify(dict_return)
