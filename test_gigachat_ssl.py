import requests

url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
bundle = "/home/kristina/certs-gigachat/gigachat_ca_bundle.pem"

try:
    r = requests.get(url, verify=bundle, timeout=20)
    print("status:", r.status_code)
    print("text:", r.text[:300])
except Exception as e:
    print(type(e).__name__)
    print(e)
