import requests
import json

url = "https://xgb-andes-endpoint.azurewebsites.net/xgb_andes"

payload = json.dumps([
  {
    "Year": 2014,
    "Mileage": 35436,
    "State": " TX",
    "Make": "Ford",
    "Model": "F-150STX"
  },
  {
    "Year": 2013,
    "Mileage": 91812,
    "State": " NE",
    "Make": "Chevrolet",
    "Model": "EquinoxFWD"
  }
])
headers = {
  'Content-Type': 'application/json',
  'Cookie': 'ARRAffinity=e3b90f2f25be5ab9fc273723a4dcfce6fc428655385e831ccea91ab3a74e0f6f; ARRAffinitySameSite=e3b90f2f25be5ab9fc273723a4dcfce6fc428655385e831ccea91ab3a74e0f6f'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
