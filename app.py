from flask import Flask
from model import *

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "darrylisveryhandsome"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.model = loadModel()

if __name__ == '__main__':
    app.run(debug=True)
