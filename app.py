from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Store user-specific model
model = None

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/train", methods=["POST"])
def train():
    global model
    
    try:
        # Get user input (past months & expenses)
        months = list(map(int, request.form["months"].split(",")))
        expenses = list(map(float, request.form["expenses"].split(",")))

        # Convert to DataFrame
        df = pd.DataFrame({"Month": months, "Expense": expenses})

        # Train model
        X = df[["Month"]]
        y = df["Expense"]
        model = LinearRegression()
        model.fit(X, y)

        return render_template("index.html", prediction="✅ Model trained successfully! Now enter a month to predict.")

    except Exception as e:
        return render_template("index.html", prediction=f"❌ Error in training: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    
    if model is None:
        return render_template("index.html", prediction="⚠️ Please train the model first!")

    try:
        # Get user input (month for prediction)
        month = int(request.form["month"])
        prediction = model.predict(np.array([[month]]))[0]

        return render_template("index.html", prediction=f"📈 Predicted Expense: ₹{prediction:.2f}")

    except Exception as e:
        return render_template("index.html", prediction=f"❌ Error in prediction: {e}")

if __name__ == "__main__":
    app.run()
