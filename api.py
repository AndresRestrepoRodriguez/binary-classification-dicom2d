import base64
import io
from flask import Flask, jsonify, request
from src.utils.model import read_pytorch_model_eval
from src.models.predict_model import predict_model
from src.utils.data import (
    read_yaml
)
from src.utils.schemas import BinaryPytorchModel
import os


app = Flask(__name__)
MODELS_DIRECTORY = 'models'
MODEL_CONFIG_PATH = 'models/model.yaml'

#Download model
model_cfg = read_yaml(MODEL_CONFIG_PATH)
model_structure = BinaryPytorchModel(**model_cfg)

print(model_structure)

destination_file = os.path.join(MODELS_DIRECTORY, 'tmp_model.' + model_structure.extension)

#download_public_google_drive_file(file_id=model_structure.id_file,
#                                  destination=destination_file)

#download_file_from_google_drive(file_id=model_structure.id_file,
#                                destination=destination_file)
#print(destination_file)
#decompress_file(destination_file, os.path.join(MODELS_DIRECTORY,
#                                               model_structure.folder))
#os.remove(destination_file)
#Define path of the model
#model_path = os.path.join(MODELS_DIRECTORY, model_structure.folder, model_structure.file_name)
model_path = 'models/best_model.pth'
model = read_pytorch_model_eval(weights=model_path)


@app.route('/predict', methods=['POST'])
def predict():
    global model
    data = request.get_json()
    img_data = data['image']
    #print(img_data)
    #img = io.BytesIO(base64.b64decode(data['image']))
    #try:
    prediction = predict_model(image=img_data,
                            model=model,
                            image_base64=True)
    #except Exception:
    #    return 'Failed to get prediction', 500
    #print(prediction)
    #print(type(prediction))
    
    return jsonify({'predictions': prediction})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    