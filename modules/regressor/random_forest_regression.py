import streamlit as st
from sklearn.ensemble import RandomForestRegressor

def random_forest_regressor():
    col1,col2=st.columns(2)
    with col1:
        n_estimators = st.slider("Number of trees:", 1, 100, 10, 1, key="rfr_n_estimators")
        max_features = st.selectbox("Max features:", ["auto", "sqrt", "log2"], 0, key="rfr_max_features")
        min_samples_split = st.slider("Min. samples split:", 2, 10, 2, 1, key="rfr_min_samples_split")

    with col2:
        max_depth = st.slider("Max depth:", 1, 100, 10, 1, key="rfr_max_depth")
        random_state = st.number_input("Random state:", 0, 1000000, 0, key="rfr_random_state")
        min_samples_leaf = st.slider("Min. samples leaf:", 1, 10, 1, 1, key="rfr_min_samples_leaf")
    try:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      random_state=random_state)
    except Exception as e:
        print("Error during model instantiation:", e)
        # Handle the exception here

    return model
