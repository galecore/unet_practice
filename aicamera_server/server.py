from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from nets.process import process_photo
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        if 'photo' not in request.files:
            return render_template('error.html', message='No file part')

        file = request.files['photo']
        if file.filename == '':
            return render_template('error.html', message='No filename')

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        transformed_filename, pred_filename, emph_filename = process_photo(filepath)

        return render_template('prediction.html', initial=transformed_filename, pred=pred_filename, emph=emph_filename)
    return render_template('index.html')

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=1337)
