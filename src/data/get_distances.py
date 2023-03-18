import requests
from dotenv import dotenv_values
import json

config = dotenv_values("../../.env")
API_KEY = config["MAPS_API_KEY"]


def find_dist(loc1: str, loc2: str) -> float:
    loc1 = loc1.replace(" ", "%20")
    loc2 = loc2.replace(" ", "%20")
    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={loc1}&destinations={loc2}&units=imperial&key={API_KEY}"
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    res = response.json()
    return float(res['rows'][0]["elements"][0]["distance"]["text"].split()[0])


def find_dist_to_GSU(loc1: str) -> float:
    gsu_location = "775 Commonwealth Ave, Boston, MA"
    return find_dist(loc1, loc2=gsu_location)
