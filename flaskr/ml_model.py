import os
import pickle

from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, app
)
from werkzeug.exceptions import abort

from flaskr.auth import login_required
from flaskr.db import get_db
from werkzeug.utils import secure_filename
from . import model_training

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

bp = Blueprint('ml_model', __name__)


@bp.route('/')
def index():
    return render_template('mlmodel/index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/train', methods=('GET', 'POST'))
@login_required
def train():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'train_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['train_file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # folder_ = app.config['UPLOAD_FOLDER']
        folder_ = '.'
        path_name = os.path.join(folder_, filename)
        file.save(path_name)
        model_training.train.train_model(path_name)
        flash('New model trained')

        # return path_name

    return render_template('mlmodel/index.html')


@bp.route('/evaluate', methods=('GET', 'POST'))
@login_required
def evaluate():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'evaluate_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['evaluate_file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # folder_ = app.config['UPLOAD_FOLDER']
        folder_ = '.'
        path_name = os.path.join(folder_, filename)
        file.save(path_name)

        model = pickle.load(open("model.pkl", "rb"))
        index2word_set = pickle.load(open("index2word_set.pkl", "rb"))
        mg = pickle.load(open("mg.pkl", "rb"))
        lsh = pickle.load(open("lsh.pkl", "rb"))

        messages = model_training.train.evaluate_file(path_name, model, index2word_set, mg, lsh)
        return render_template('mlmodel/results.html', messages=messages)

        # return path_name

    return render_template('mlmodel/results.html')
