import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
import time
import pandas as pd


def hyperparameter_optimization(X_train, y_train):
    st.subheader("Hyperparameter Optimization Settings")
    n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=10, step=1)
    cv = st.number_input("Number of cross-validation folds", min_value=2, value=5, step=1)
    random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

    st.write('#')

    if "ridge_best_param" not in st.session_state:
        st.session_state.ridge_best_param = {
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "auto",
            "max_iter": 1000,
            "tol": 0.0001,
            "random_state": 0
        }

    if st.button('Run Optimization'):
        st.write('#')
        param_dist = {
            "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
            "fit_intercept": [True, False],
            "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            "max_iter": [100, 500, 1000],
            "tol": [0.0001, 0.001, 0.01, 0.1, 1],
            "random_state": [0]
        }
        model = Ridge()

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

        st.session_state.ridge_best_param = best_params

    return st.session_state.ridge_best_param


def ridge_regression(X_train, y_train):
    best_params = hyperparameter_optimization(X_train, y_train)

    alpha = st.number_input("Alpha", 0.0, 100.0, key="rr_alpha",
                            value=float(best_params.get('alpha', 1.0)))
    fit_intercept = st.checkbox("Fit Intercept", key="rr_fit_intercept",
                                value=best_params.get('fit_intercept', True))
    max_iter = st.number_input("Max Iterations", 1, 100000, key="rr_max_iter",
                               value=best_params.get('max_iter', 1000))
    solver = st.selectbox("Solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                          key="rr_solver",
                          index=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"].index(best_params.get('solver', 'auto')))
    tol = st.number_input("Tolerance", 1e-6, 1.0, key="rr_tol",
                          value=float(best_params.get('tol', 0.0001)))

    try:
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter,
                      solver=solver, tol=tol, random_state=best_params.get('random_state', 0))
    except ValueError as e:
        print(f"Error creating Ridge model: {e}")
        model = None

    return model
