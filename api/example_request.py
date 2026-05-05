"""
Example: How to send a request to the Perovskite Classifier API
Run the API server first: uvicorn api.app:app --reload
"""

import requests

API_URL = "http://localhost:8000/predict"

# Sample compound (similar to AgBiO3 in the dataset)
sample_input = {
    "A_OS": 1,
    "A_prime_OS": 1,
    "A_HOMO_minus": -5.6,
    "A_HOMO_plus": -5.6,
    "A_IE_minus": 7.58,
    "A_IE_plus": 7.58,
    "A_LUMO_minus": -0.5,
    "A_LUMO_plus": -0.5,
    "A_X_minus": 1.93,
    "A_X_plus": 1.93,
    "A_Z_radii_minus": 1.29,
    "A_Z_radii_plus": 1.29,
    "A_e_affin_minus": 1.30,
    "A_e_affin_plus": 1.30,
    "B_OS": 5,
    "B_prime_OS": 5.0,
    "B_HOMO_minus": -7.2,
    "B_HOMO_plus": -7.2,
    "B_IE_minus": 8.0,
    "B_IE_plus": 8.0,
    "B_LUMO_minus": -1.1,
    "B_LUMO_plus": -1.1,
    "B_X_minus": 2.02,
    "B_X_plus": 2.02,
    "B_Z_radii_minus": 1.03,
    "B_Z_radii_plus": 1.03,
    "B_e_affin_minus": 0.95,
    "B_e_affin_plus": 0.95,
    "mu": 0.54,
    "tau": 0.00,
    "new_tol": 0.00,
    "t": 0.946
}

response = requests.post(API_URL, json=sample_input)

if response.status_code == 200:
    result = response.json()
    print(f"Prediction     : {result['label']}")
    print(f"P(Semiconductor): {result['probability_semiconductor']:.4f}")
    print(f"P(Metal)       : {result['probability_metal']:.4f}")
else:
    print(f"Error {response.status_code}: {response.text}")
