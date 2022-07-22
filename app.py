from crypt import methods
from flask import Flask, request 
from utils.load_output import get_output
from utils.constants import UPLOAD_DIR
import os
import flask 
import shutil
from flask import jsonify

app = Flask(__name__)


app.config["UPLOAD_FOLDER"] = UPLOAD_DIR


@app.route("/predict", methods=["GET", "POST"])
def predict():
    os.makedirs("./input_data", exist_ok=True)
    if flask.request.method == "POST":
        files = flask.request.files.getlist("file[]")
        for file in files:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    result = get_output()
    shutil.rmtree("./input_data")
    shutil.rmtree("./processed_data")
    return jsonify(result)


if __name__ == "__main__":
    app.run(port=5050, debug=True)