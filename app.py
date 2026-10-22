import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
with open("svm.pkl", "rb") as f:
  model = pickle.load(f)

# HTML template with an attractive Bootstrap 5 layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Assessment Portal</title>
    <!-- Bootstrap 5 CSS CDN -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 2rem 0;
        }
        .card {
            border: none;
            border-radius: 1rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .btn-primary {
            background: linear-gradient(45deg, #4e73df, #224abe);
            border: none;
            border-radius: 0.5rem;
            padding: 0.75rem;
            font-weight: 600;
        }
        .btn-primary:hover {
            background: linear-gradient(45deg, #224abe, #1a38b5);
        }
        .form-control, .form-select {
            border-radius: 0.5rem;
            padding: 0.65rem 1rem;
        }
        .result-box {
            background-color: #e8f4fd;
            border-left: 5px solid #3498db;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-top: 1.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card p-4 p-md-5 bg-white">
                    <div class="text-center mb-4">
                        <h2 class="fw-bold text-primary">Health Risk Predictor</h2>
                        <p class="text-muted">Enter your health metrics below to generate an instant prediction.</p>
                    </div>

                    <form method="POST">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Age</label>
                                <input type="number" class="form-control" name="age" placeholder="e.g. 45" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Gender (Encoded Code)</label>
                                <input type="number" class="form-control" name="gender" placeholder="e.g. 0 or 1" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">City Code</label>
                                <input type="number" class="form-control" name="city" placeholder="City numeric code" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">BMI</label>
                                <input type="number" step="0.1" class="form-control" name="bmi" placeholder="e.g. 24.5" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Family History of Diabetes</label>
                                <input type="number" class="form-control" name="family_history_diabetes" placeholder="0 (No) / 1 (Yes)" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Physical Activity Level</label>
                                <input type="number" class="form-control" name="physical_activity_level" placeholder="Level code" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Diet Type</label>
                                <input type="number" class="form-control" name="diet_type" placeholder="Diet code" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Smoking Status</label>
                                <input type="number" class="form-control" name="smoking_status" placeholder="Status code" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Alcohol Consumption</label>
                                <input type="number" class="form-control" name="alcohol_consumption" placeholder="Consumption code" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Hours Sleep Per Night</label>
                                <input type="number" step="0.1" class="form-control" name="hours_sleep_per_night" placeholder="e.g. 7.5" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Stress Level</label>
                                <input type="number" class="form-control" name="stress_level" placeholder="Scale 1-10" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Fasting Blood Sugar</label>
                                <input type="number" step="0.1" class="form-control" name="fasting_blood_sugar" placeholder="e.g. 95.0" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">HbA1c Level</label>
                                <input type="number" step="0.1" class="form-control" name="hba1c_level" placeholder="e.g. 5.6" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Blood Pressure (Systolic)</label>
                                <input type="number" class="form-control" name="blood_pressure_systolic" placeholder="e.g. 120" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Blood Pressure (Diastolic)</label>
                                <input type="number" class="form-control" name="blood_pressure_diastolic" placeholder="e.g. 80" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Waist Circumference (cm)</label>
                                <input type="number" step="0.1" class="form-control" name="waist_circumference_cm" placeholder="e.g. 85.0" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Income Bracket</label>
                                <input type="number" class="form-control" name="income_bracket" placeholder="Bracket code" required>
                            </div>
                        </div>

                        <div class="mt-4 text-center">
                            <button type="submit" class="btn btn-primary w-100 py-2">Run Prediction</button>
                        </div>
                    </form>

                    {% if prediction is not none %}
                    <div class="result-box text-center">
                        <h4 class="fw-bold text-dark mb-2">Analysis Result</h4>
                        <p class="fs-5 mb-0">Outcome: <span class="badge bg-success fs-6">{{ prediction }}</span></p>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
  prediction = None
  if request.method == "POST":
    try:
      # Collecting features in the exact order parsed from model metadata
      features = [
          float(request.form["age"]),
          float(request.form["gender"]),
          float(request.form["city"]),
          float(request.form["bmi"]),
          float(request.form["family_history_diabetes"]),
          float(request.form["physical_activity_level"]),
          float(request.form["diet_type"]),
          float(request.form["smoking_status"]),
          float(request.form["alcohol_consumption"]),
          float(request.form["hours_sleep_per_night"]),
          float(request.form["stress_level"]),
          float(request.form["fasting_blood_sugar"]),
          float(request.form["hba1c_level"]),
          float(request.form["blood_pressure_systolic"]),
          float(request.form["blood_pressure_diastolic"]),
          float(request.form["waist_circumference_cm"]),
          float(request.form["income_bracket"]),
      ]

      pred = model.predict([features])[0]
      prediction = (
          "High Risk / Positive"
          if str(pred) == "1"
          else "Low Risk / Negative"
      )
    except Exception as e:
      prediction = f"Error processing input: {str(e)}"

  return render_template_string(HTML_TEMPLATE, prediction=prediction)


if __name__ == "__main__":
  app.run(debug=True)
