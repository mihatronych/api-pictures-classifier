from flask import Flask, jsonify
from flask import request
from flask import abort
from flask_restful import Api
from waitress import serve
import pictures

app = Flask(__name__)
api = Api(app)


@app.route('/toxicity_py/api/picture', methods=['POST','GET'])
def get_picture():
    if request.method == 'GET':
        #file = request.files['image'].read()  ## byte file
        #print(file)
        #npimg = np.fromstring(file, np.uint8)
        url = request.json['url']
        untoxic, toxic = pictures.classify_pic(url)
        result = []
        result.append({
                'untoxic': str(untoxic),
                'toxic': str(toxic)
            })
        return jsonify(result)
    else:
        abort(400)

@app.route('/toxicity_py/api/picture_text', methods=['POST','GET'])
def get_picture_text():
    if request.method == 'GET':
        #file = request.files['image'].read()  ## byte file
        #print(file)
        #npimg = np.fromstring(file, np.uint8)
        url = request.json['url']
        text = pictures.scan_pic(url)
        return jsonify({
                'text': str(text),
            })
    else:
        abort(400)

if __name__ == "__main__":
    serve(app, host="localhost", port=7000)
