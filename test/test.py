import requests
import base64

with open('data/0717d112-a8e5-46c0-9ba5-902cd9fe8fe1.dcm', 'rb') as f:
    encoded_image = base64.b64encode(f.read()).decode('utf-8')

data = {'image': encoded_image}
response = requests.post('http://127.0.0.1:5000/predict', json=data)
print(response.content)