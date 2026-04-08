import os
import time

import pandas as pd
from flask import Flask, jsonify, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash, generate_password_hash

DEFAULT_ITEMS_PER_PAGE = 100

def _read_cars(file_path):
    try:
        cars = pd.read_csv(file_path)
        if cars.empty:
            raise ValueError("The cars DataFrame is empty.")
        # Optional: convert column names to snake_case for consistency (not required, but cleaner)
        cars.columns = [col.strip().replace(" ", "_").replace("(", "").replace(")", "") for col in cars.columns]
        return cars
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    except Exception as e:
        raise e


app = Flask(__name__)
app.config["cars"] = _read_cars("/cars.csv")

auth = HTTPBasicAuth()
# users = {
#         os.environ.get("API_USER", "airflow"): generate_password_hash(
#         os.environ.get("API_PASSWORD", "airflow"), method="sha256"
#     )
# }

users = { "airflow": generate_password_hash("airflow", method="pbkdf2")}

@auth.verify_password
def verify_password(username, password):
    if username in users:
        return check_password_hash(users.get(username), password)
    return False


@app.route("/")
def hello():
    return "Hello from the Car Data API!"


@app.route("/cars")
@auth.login_required
def cars():
    """
    Returns car records with optional filtering and pagination.

    Query Parameters
    ----------------
    min_year : int
        Minimum year (inclusive).
    max_year : int
        Maximum year (inclusive).
    min_price : float
        Minimum price (inclusive).
    max_price : float
        Maximum price (inclusive).
    fuel_type : int
        type code (e.g., 0=petrol, 1=diesel, 2=electric, etc.).
    transmission : int
        Transmission type: 0=auto, 1=manual.
    make : str
        Partial or full match for Make (case-insensitive).
    model : str
        Partial or full match for Model (case-insensitive).
    offset : int
        Pagination offset (default: 0).
    limit : int
        Max number of results (default: 100).
    """
    cars_df = app.config["cars"]

    # Apply filters
    if "min_year" in request.args:
        min_y = int(request.args["min_year"])
        cars_df = cars_df[cars_df["Year"] >= min_y]

    if "max_year" in request.args:
        max_y = int(request.args["max_year"])
        cars_df = cars_df[cars_df["Year"] <= max_y]

    if "min_price" in request.args:
        min_p = float(request.args["min_price"])
        cars_df = cars_df[cars_df["Price_euro"] >= min_p]

    if "max_price" in request.args:
        max_p = float(request.args["max_price"])
        cars_df = cars_df[cars_df["Price_euro"] <= max_p]

    if "fuel_type" in request.args:
        ft = int(request.args["fuel_type"])
        cars_df = cars_df[cars_df["Fuel_type"] == ft]

    if "transmission" in request.args:
        tr = int(request.args["transmission"])
        cars_df = cars_df[cars_df["Transmission"] == tr]

    if "make" in request.args:
        make_query = request.args["make"].lower()
        cars_df = cars_df[cars_df["Make"].astype(str).str.lower().str.contains(make_query)]

    if "model" in request.args:
        model_query = request.args["model"].lower()
        cars_df = cars_df[cars_df["Model"].astype(str).str.lower().str.contains(model_query)]

    # Pagination
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", DEFAULT_ITEMS_PER_PAGE))

    subset = cars_df.iloc[offset : offset + limit]

    return jsonify(
        {
            "result": subset.to_dict(orient="records"),
            "offset": offset,
            "limit": limit,
            "total": len(cars_df),
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)