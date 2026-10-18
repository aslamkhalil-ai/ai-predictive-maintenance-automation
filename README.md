
# AI Predictive Maintenance and Intelligent Automation Platform

An end-to-end machine learning and automation project that predicts potential industrial machine failures using sensor data and triggers automated actions through n8n.

## Project Overview

This project combines Machine Learning, Flask REST API development, and workflow automation to demonstrate how predictive maintenance systems can support industrial operations.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Classifier
- Flask REST API
- n8n Workflow Automation
- Google Sheets
- Gmail
- Google Colab

## System Architecture

1. Train a machine failure prediction model using the AI4I 2020 Predictive Maintenance dataset.
2. Save the trained model using Joblib.
3. Load the model into a Flask API.
4. Send sensor data to the API through an HTTP request.
5. Use n8n to route predictions based on the model output.
6. Log normal predictions in Google Sheets.
7. Send Gmail alerts when a machine failure is predicted.

## Input Features

- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/predict` | Predict machine failure |

## Example Prediction Request

```json
{
  "type": "M",
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551,
  "torque": 42.8,
  "tool_wear": 120
}
```

## Local Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Ensure the trained model exists at:

   `models/predictive_maintenance_model.joblib`

5. Start the Flask API:

   `python app.py`

6. Open `http://127.0.0.1:5000/`.

## Automation Workflow

The n8n workflow connects the Flask prediction API to conditional routing, Google Sheets logging, and Gmail notifications.

## Important Disclaimer

Predictions are model estimates, not guarantees of machine failure. This project is intended for educational and portfolio demonstration purposes, not as a certified industrial safety system.

## Author

Muhammad Aslam Khalil
