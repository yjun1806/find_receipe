import os
import Load_Saved_Model
import make_json

from flask import Flask, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    print("1 Something is coming!!")
    if request.method == 'POST':
        if 'upload' not in request.files:
            print('No upload!!')
            return 'bad'
        f = request.files['upload']
        print(f)
        img_path = os.path.join('/home/kunde/Capstone_Project/upload_img/', secure_filename(f.filename))
        f.save(img_path)

        print('Save path : ' + img_path)
        result, classes = Load_Saved_Model.predict(img_path)
        json_data = make_json.set_data(result, classes)
        print(json_data)

    return json_data


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)


