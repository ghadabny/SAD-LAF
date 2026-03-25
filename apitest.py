import requests
url = "https://proxy.transport.data.gouv.fr/resource/sncf-gtfs-rt-trip-updates"
# Sans proxy, via partage de connexion téléphone
response = requests.get(url, timeout=30)
print("Status :", response.status_code)
print("Taille :", len(response.content))