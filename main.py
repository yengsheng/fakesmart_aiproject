import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from model import *

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def filterOptions():
    filterPath = app.static_folder + "/filter"
    filterOption = []
    for filename in os.listdir(filterPath):
        if os.path.isfile(os.path.join(filterPath, filename)):
            filterOption.append({'name': filename})
    return filterOption

@app.route('/')
def upload_form():
    return render_template('website.html', data=filterOptions())

@app.route('/', methods=['POST'])
def generate():
    if 'file' not in request.files:
        flash('No file path')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == "":
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filtername = request.form.get('selected')
        inputname = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], inputname))
        flash('Image successfully uploaded!')

        # TODO
        ## function to generate confidence score for original image
        confidence, classification = prediction(os.path.join(app.config['UPLOAD_FOLDER'], inputname), app.model)
        # function to generate confidence score for new image with filter

        outputname = "sample.jpg"

        inputData = {'inputname': inputname, 'confidence': confidence, 'classification': classification}
        outputData = {'outputname': outputname, 'confidence': "sampleConf", 'classification': "sampleClass"}
        return render_template('website.html', data=filterOptions(), input=inputData, filterselected=filtername, output=outputData)
    else:
        flash('Allowed image types are: png, jpg, jpeg')
        return redirect(request.url)

@app.route('/input/<inputname>')
def input_image(inputname):
    return redirect(url_for('static', filename='uploads/' + inputname))

@app.route('/filter/<filtername>')
def filter_image(filtername):
    return redirect(url_for('static', filename='filter/' + filtername))

@app.route('/output/<outputname>')
def output_image(outputname):
    return redirect(url_for('static', filename='generated/' + outputname))

if __name__ == "__main__":
    app.run()