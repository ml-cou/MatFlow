import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.model_selection import train_test_split

def train_model(data, features, target_variables):
    # Split the data into features and targets
    X = data[features]
    y = data[target_variables]

    # Perform any necessary preprocessing on the features and targets
    # ...

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the model architecture
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(len(target_variables)))  # Output layer

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32)

    # Evaluate the model on the testing set
    loss = model.evaluate(X_test_scaled, y_test)

    # Optionally, you can save the trained model for future use
    # model.save('trained_model.h5')

    return model


def preprocess_input(input_features):
    # Convert input features to the desired data type
    input_features = [int(feature) for feature in input_features.split(",")]
    input_features = np.array(input_features).reshape(1, -1)
    return input_features

def reverse_ml(dataset, table_name):
    data = dataset[table_name]

    all_columns = data.columns.to_list()

    target_variables = st.multiselect('Select Features', data.columns.to_list(), key='target')
    for target in target_variables:
        all_columns.remove(target)

    features = st.multiselect('Select Target Variables', all_columns, key='features')

    st.write('#')

    if "model" not in st.session_state:
        st.session_state.model = None
    if 'target_pre' not in st.session_state:
        st.session_state.target_pre = None
    if 'features_pre' not in st.session_state:
        st.session_state.features_pre = None
    if "table_pre" not in st.session_state:
        st.session_state.table_pre = None

    if st.button('Create Model', key='create_model'):
        if st.session_state.model is None or st.session_state.table_pre != table_name or st.session_state.target_pre != target_variables or st.session_state.features_pre != features:
            st.session_state.model = train_model(data, features, target_variables)
            st.session_state.target_pre = target_variables
            st.session_state.features_pre = features
            st.session_state.table_pre = table_name

    if st.session_state.model:
        st.title('Prediction for All Target Variables')

        input_features = st.text_input(f'Enter values for {features}', key='input')
        st.write('#')

        if st.button('Predict', key='predict'):
            input_features = preprocess_input(input_features)
            prediction = st.session_state.model.predict([input_features])[0]
            prediction = prediction.reshape(1, -1)

            st.subheader('Prediction:')
            prediction_table = pd.DataFrame(
                {'Target Variable': target_variables, 'Predicted Value': prediction.flatten()})

            st.table(prediction_table)
