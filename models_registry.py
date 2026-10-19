"""models_registry.py -- name -> build_model() for every algorithm in the paper."""
import alg01_linear_regression as a1
import alg02_decision_tree as a2
import alg03_random_forest as a3
import alg04_gradient_boosting as a4
import alg05_xgboost as a5
import alg06_ffnn as a6
import alg07_cnn as a7
import alg08_lstm as a8
import alg09_homogeneous_stacking as a9
import alg10_ag_stacknet as a10

MODELS = {
    "Linear Regression": a1.build_model,
    "Decision Tree": a2.build_model,
    "Random Forest": a3.build_model,
    "Gradient Boosting": a4.build_model,
    "XGBoost": a5.build_model,
    "FFNN": a6.build_model,
    "CNN": a7.build_model,
    "LSTM": a8.build_model,
    "Homogeneous Stacking": a9.build_model,
    "AG-StackNet": a10.build_model,
}
