import os
from flask import Flask, flash, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.shared_data import SharedDataMiddleware

from app import ner_trainer
from app import pdf_extractor
from app.tools import upload_file_to_s3


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})


def cvparser(filepath):
    content = pdf_extractor.extract_pdf_content(filepath)
    model_filepath = os.getcwd() + "/models/model_ner_53"

    return ner_trainer.predict_spacy(content, model_filepath)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.filename = secure_filename(file.filename)
            filepath = upload_file_to_s3(file, os.environ['AWS_S3_BUCKET'])
            resp = cvparser(filepath)
            return jsonify(resp)
        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #     file.save(filepath)
        #     resp = cvparser(filepath)
        #     return jsonify(resp)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
