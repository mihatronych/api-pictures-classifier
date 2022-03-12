from flask import Flask, jsonify
from flask import request
from flask import abort
from flask_restful import Api
from waitress import serve
import pictures

app = Flask(__name__)
api = Api(app)
import numpy as np

@app.route('/toxicity_py/api/picture', methods=['POST', 'GET'])
def get_picture():
    if request.method == 'POST':
        file = request.files['image'].read()  ## byte file
        print(file)
        npimg = np.fromstring(file, np.uint8)
        untoxic, toxic = pictures.classify_pic(npimg)
        result = []
        result.append({
                'untoxic': str(untoxic),
                'toxic': str(toxic)
            })
        return jsonify(result)
    else:
        abort(400)

@app.route('/toxicity_py/api/pictures', methods=['POST', 'GET'])
def get_pictures():
    if request.method == 'POST':
        file = request.files  ## byte file
        result = []
        i = 0
        for f in file:
            i += 1
            npimg = np.fromstring(f,  np.uint8)
            untoxic, toxic = pictures.classify_pic(npimg)
            result.append({i: {
                'untoxic': str(untoxic),
                'toxic': str(toxic)
            }})
        return jsonify(result)
    else:
        abort(400)


if __name__ == "__main__":
    serve(app, host="localhost", port=7000)
