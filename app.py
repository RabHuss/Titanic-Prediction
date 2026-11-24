
import os
import pickle
import pandas as pd

from flask import Flask, render_template, request

app = Flask(
    __name__,
    template_folder="templates"
)

# ----------------------------------
# 1. MODEL DIRECTORY
# ----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("Project folder:", BASE_DIR)
print("Models folder:", MODEL_DIR)
print("Files inside models folder:", os.listdir(MODEL_DIR))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)

    with open(path, "rb") as file:
        return pickle.load(file)


# ----------------------------------
# 2. LOAD ALL FOUR MODELS
# ----------------------------------

survival_model = load_model("titanic_survival_model.pkl")
family_model = load_model("titanic_family_status_model.pkl")
age_model = load_model("titanic_age_model.pkl")
pclass_model = load_model("titanic_pclass_model.pkl")


# ----------------------------------
# 3. HOME PAGE
# ----------------------------------

models = [
    {
        "id": "survival",
        "title": "Survival Prediction",
        "description": "Predict whether a Titanic passenger survived.",
        "icon": "❤️"
    },
    {
        "id": "family_status",
        "title": "Family Status",
        "description": "Predict whether a passenger travelled alone or with family.",
        "icon": "👨‍👩‍👧‍👦"
    },
    {
        "id": "age",
        "title": "Age Prediction",
        "description": "Estimate a passenger's age.",
        "icon": "🎂"
    },
    {
        "id": "pclass",
        "title": "Passenger Class",
        "description": "Predict first, second or third class.",
        "icon": "🚢"
    }
]


@app.route("/")
def home():
    return render_template("index.html", models=models)


# ----------------------------------
# 4. SURVIVAL PREDICTION
# ----------------------------------

@app.route("/survival", methods=["GET", "POST"])
def survival():

    result = None

    if request.method == "POST":

        age = float(request.form["age"])
        pclass = int(request.form["pclass"])
        sibsp = int(request.form["sibsp"])
        parch = int(request.form["parch"])
        fare = float(request.form["fare"])
        sex = request.form["sex"]
        embarked = request.form["embarked"]

        # Match the exact training columns and order
        passenger = pd.DataFrame([{
            "Age": age,
            "Pclass": pclass,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Sex_male": 1 if sex == "male" else 0,
            "Embarked_Q": 1 if embarked == "Q" else 0,
            "Embarked_S": 1 if embarked == "S" else 0
        }], columns=[
            "Age", "Pclass", "SibSp", "Parch",
            "Fare", "Sex_male", "Embarked_Q", "Embarked_S"
        ])

        prediction = survival_model.predict(passenger)[0]

        result = (
            "Likely to Survive"
            if prediction == 1
            else "Likely Not to Survive"
        )

    return render_template("survival.html", result=result)


# ----------------------------------
# 5. FAMILY STATUS PREDICTION
# ----------------------------------

@app.route("/family-status", methods=["GET", "POST"])
def family_status():

    result = None

    if request.method == "POST":

        passenger = pd.DataFrame([{
            "Sex": request.form["sex"],
            "Pclass": int(request.form["pclass"]),
            "Age": float(request.form["age"]),
            "Fare": float(request.form["fare"]),
            "Embarked": request.form["embarked"]
        }])

        result = family_model.predict(passenger)[0]

    return render_template("family_status.html", result=result)


# ----------------------------------
# 6. AGE PREDICTION
# ----------------------------------

@app.route("/age", methods=["GET", "POST"])
def age():

    result = None

    if request.method == "POST":

        passenger = pd.DataFrame([{
            "Sex": request.form["sex"],
            "Pclass": int(request.form["pclass"]),
            "Fare": float(request.form["fare"]),
            "SibSp": int(request.form["sibsp"]),
            "Parch": int(request.form["parch"])
        }])

        prediction = age_model.predict(passenger)[0]

        result = round(float(prediction), 1)

    return render_template("age.html", result=result)


# ----------------------------------
# 7. PASSENGER CLASS PREDICTION
# ----------------------------------

@app.route("/pclass", methods=["GET", "POST"])
def pclass():

    result = None

    if request.method == "POST":

        passenger = pd.DataFrame([{
            "Sex": request.form["sex"],
            "Age": float(request.form["age"]),
            "Fare": float(request.form["fare"]),
            "SibSp": int(request.form["sibsp"]),
            "Parch": int(request.form["parch"]),
            "Embarked": request.form["embarked"]
        }])

        prediction = pclass_model.predict(passenger)[0]

        class_names = {
            1: "First Class",
            2: "Second Class",
            3: "Third Class"
        }

        result = class_names[int(prediction)]

    return render_template("pclass.html", result=result)


# ----------------------------------
# 8. RUN APPLICATION
# ----------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)

    print("All four Titanic models loaded successfully!")
print("Starting Titanic Prediction Website...")
