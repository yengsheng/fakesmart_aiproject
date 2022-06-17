from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    pass

if __name__ == '__main__':
    app.run(debug=True)
