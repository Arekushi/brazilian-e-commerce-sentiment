from dotenv import load_dotenv
from os import getenv
from os.path import join, dirname, isfile


from apps import create_app


_ENV_FILE = join(dirname(__file__), '.env')
if isfile(_ENV_FILE):
    load_dotenv(dotenv_path=_ENV_FILE)


app = create_app(getenv('FLASK_ENV') or 'default')


if __name__ == '__main__':
    ip = 'localhost'
    port = app.config['APP_PORT']
    debug = app.config['DEBUG']
    app.run(
        host=ip, debug=debug, port=port, use_reloader=debug
    )
