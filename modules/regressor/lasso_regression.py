import streamlit as st
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import time

def hyperparameter_optimization(X_train, y_train):
    st.subheader("Hyperparameter Optimization Settings")
    n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=10, step=1)
    cv = st.number_input("Number of cross-validation folds", min_value=2, value=5, step=1)
    random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

    st.write('#')

    if "lasso_best_param" not in st.session_state:
        st.session_state.lasso_best_param = {
            "alpha": 1.0,
            "fit_intercept": True,
            "max_iter": 1000,
            "positive": False,
            "selection": "cyclic",
            "tol": 0.0001,
            "warm_start": False
        }

    if st.button('Run Optimization'):
        st.write('#')
        param_dist = {
            "alpha": [0.001, 0.01, 0.1, 1, 10],
            "fit_intercept": [True, False],
            "max_iter": [1000, 5000, 10000],
            "positive": [True, False],
            "selection": ["cyclic", "random"],
            "tol": [0.0001, 0.001, 0.01, 0.1],
            "warm_start": [True, False]
        }
        model = Lasso()

        with st.spinner('Doing hyperparameter optimization...'):
            for i in range(100):
                time.sleep(0.1)
                st.spinner(f"Running iteration {i + 1} of {n_iter}...")
            clf = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv,
                                     random_state=random_state)
            st.spinner("Fitting the model...")
            clf.fit(X_train, y_train)
        st.success("Hyperparameter optimization completed!")
        best_params = clf.best_params_

        st.write('Best estimator:')
        rows = []
        for param, value in best_params.items():
            rows.append([param, value])
        st.table(pd.DataFrame(rows, columns=['Parameter', 'Value']))

        st.session_state.opti_results_df = pd.DataFrame(rows, columns=['Parameter', 'Value'])

        st.session_state.lasso_best_param = best_params

    return st.session_state.lasso_best_param

def lasso_regression(X_train, y_train):
    best_params = hyperparameter_optimization(X_train, y_train)


    alpha = st.number_input(
        "Alpha", 0.0, 100.0, key="ls_alpha",
        value=float(best_params.get('alpha', 1.0))
    )
    fit_intercept = st.checkbox(
        "Fit Intercept", key="ls_fit_intercept",
        value=best_params.get('fit_intercept', True)
    )
    normalize = st.checkbox("Normalize", False, key="ls_normalize")
    max_iter = st.number_input(
        "Max Iterations", 1, 100000, key="ls_max_iter",
        value=int(best_params.get('max_iter', 1000))
    )
    selection = st.selectbox(
        "Selection", ["cyclic", "random"], key="ls_selection",
        index=["cyclic", "random"].index(best_params.get('selection', 'cyclic'))
    )
    tol = st.number_input(
        "Tolerance", 1e-6, 1e-1, key="ls_tol",
        value=float(best_params.get('tol', 0.0001))
    )
    warm_start = st.checkbox("Warm Start", False, key="ls_warm_start")

    try:
        model = Lasso(
            alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, max_iter=max_iter,
            selection=selection, tol=tol, warm_start=warm_start
        )
    except ValueError as e:
        # Handle any ValueErrors that occur during initialization
        print("Error: ", e)
        model = None

    return model
