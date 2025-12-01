from flask import Flask, render_template, request
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load models
model_tree = joblib.load("decision_tree_learning_path.pkl")
model_log = joblib.load("logreg_ready.pkl")
model_rf = joblib.load("rf_difficulty.pkl")
model_svc = joblib.load("svc_ready.pkl")
model_perf = joblib.load("elasticnet_performance.pkl")

FEATURES = [
    'prior_knowledge_score','avg_quiz_score','cognitive_load_score','avg_time_per_question',
    'error_rate','revision_frequency','help_requests_count','attention_span_score',
    'mental_fatigue_score','time_variation','concept_mastery_rate','engagement_rate',
    'avg_study_time_per_day'
]

@app.route("/")
def index():
    return render_template("index.html", features=FEATURES)

def preprocess(form):
    vals = [float(form[f]) for f in FEATURES]
    return np.array(vals).reshape(1, -1)

@app.route("/predict", methods=["POST"])
def predict():
    x = preprocess(request.form)

    path = int(model_tree.predict(x)[0])
    ready_log = bool(model_log.predict(x)[0])
    ready_svc = bool(model_svc.predict(x)[0])
    diff = int(model_rf.predict(x)[0])
    perf = float(model_perf.predict(x)[0])

    # Maps
    path_map = {0:"Slow",1:"Moderate",2:"Fast"}
    diff_map = {0:"Easy",1:"Medium",2:"Hard"}

    # simple agreement logic
    agree = f"{int(ready_log==ready_svc)}/1 models agree"

    return render_template(
        "result.html",
        performance=round(perf,2),
        path=path_map[path],
        diff=diff_map[diff],
        ready_log=ready_log,
        ready_svc=ready_svc,
        agree=agree
    )

if __name__ == "__main__":
    app.run(debug=True)
