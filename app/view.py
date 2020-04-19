import os
import logging
from flask import Blueprint, flash, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from app import CVParser
from app.tools import upload_file_to_s3

load_dotenv()

bp = Blueprint("v1", __name__, template_folder=os.path.join(os.getcwd(), 'app/templates'))

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
parser = CVParser()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        logging.info(file.filename)
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.filename = secure_filename(file.filename)
            upload_file_to_s3(file, os.getenv('AWS_S3_BUCKET'))
            filepath = "https://{}.s3.{}.amazonaws.com/{}".format(
                os.getenv('AWS_S3_BUCKET'),
                os.getenv('AWS_REGION'),
                file.filename
            )
            resp = parser.parse_cv(filepath)
            return jsonify(resp)
        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #     file.save(filepath)
        #     resp = parser.parse_cv(filepath)
        #     return jsonify(resp)

    return render_template('upload.html')
