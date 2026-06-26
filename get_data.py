"""
get_data.py
------------
Downloads the Telco Customer Churn dataset (7,043 customers) into data/telco.csv.

Run it with:   python get_data.py
"""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
os.makedirs("data", exist_ok=True)
OUT = "data/telco.csv"

print("Downloading Telco Customer Churn dataset...")
urllib.request.urlretrieve(URL, OUT)
print(f"Saved to {OUT}")
