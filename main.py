import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from model import *
from utils import *

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def filterOptions():
    list = []
    for key in attOptions.keys():
        list.append({'name': key})
    return list

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
        confidence, classification, image_forward = prediction(os.path.join(app.config['UPLOAD_FOLDER'], inputname), app.model)

        # function to generate confidence score for new image with filter
        confidence_a, classification_a, generated_name, noise_name = attack(inputname.split(".")[0], image_forward, app.model, classes[classification], filtername)

        if confidence_a == 0 and classification_a == 0:
            inputData = {'inputname': inputname, 'confidence': confidence, 'classification': classes[classification]}
            outputData = {'outputname': 'failed', 'confidence': 'failed', 'classification': 'failed'}
            attackData = {'attackSelected': 'failed', 'noise': 'failed'}
        else:
            inputData = {'inputname': inputname, 'confidence': confidence, 'classification': classes[classification]}
            outputData = {'outputname': generated_name, 'confidence': confidence_a, 'classification': classes[classification_a]}
            attackData = {'attackSelected': filtername, 'noise': noise_name}
        return render_template('website.html', data=filterOptions(), input=inputData, attack=attackData, output=outputData)
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