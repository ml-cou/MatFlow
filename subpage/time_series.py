import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta


def time_series_analysis(dataset, table_name):
    if 'save_as' not in st.session_state:
        st.session_state.save_as = True

    try:
        data = dataset[table_name]
        data_opt = table_name
    except KeyError:
        st.header("No Dataset Found")
        st.stop()
    except Exception as e:
        st.warning(e)
        st.stop()

    st.subheader("Time Series Analysis")

    date_columns = []

    for column_name in data.columns:
        if data[column_name].dtype == 'object' or data[column_name].dtype == 'datetime64[ns]':
            try:
                pd.to_datetime(data[column_name])
                date_columns.append(column_name)
            except ValueError:
                pass

    # Display the dataset
    st.write("Original Dataset:")
    st.write(data)
    if len(date_columns) == 0:
        st.warning(
            "No datetime columns found in the dataset. Please check your data or specify the correct column types.")
        return
    data.rename(columns={date_columns[0]:"date"},inplace=True)
    data['date'] = pd.to_datetime(data["date"])

    data = data.set_index('date')

    # Define the time range slider
    min_date_range = data.index.min() - timedelta(days=365 * 20)
    max_date_range = data.index.max() + timedelta(days=365 * 20)

    selected_range = st.slider("Select a time range:", min_value=min_date_range, max_value=max_date_range,
                               value=(data.index.min().to_pydatetime(), data.index.max().to_pydatetime()))

    # Convert selected range to Timestamp objects
    selected_start = pd.Timestamp(selected_range[0])
    selected_end = pd.Timestamp(selected_range[1])

    # Filter the data based on the selected range
    filtered_data = data.loc[selected_start:selected_end]
    # Plot the time series
    st.write("Time Series Plot:")
    if isinstance(filtered_data, pd.DataFrame):
        for column in filtered_data.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(filtered_data[column])
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.title(f"Time Series Plot - {column}")

            # Perform forecasting
            model = ARIMA(filtered_data[column], order=(1, 1, 1))
            model_fit = model.fit()

            # Define the prediction range
            prediction_start = selected_start
            prediction_end = selected_end    #prediction_start + timedelta(days=365 * 10)  # Predict for 10 years (1990)

            forecast = model_fit.predict(start=prediction_start, end=prediction_end)

            # Plot the forecasted values
            plt.plot(forecast.index, forecast.values, label="Forecasted")
            plt.legend()

            # Set the x-axis limits for data and forecast
            plt.xlim(selected_start, prediction_end)

            st.pyplot(plt)
    elif isinstance(filtered_data, pd.Series):
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_data)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Time Series Plot")

        # Perform forecasting
        model = ARIMA(filtered_data, order=(1, 1, 1))
        model_fit = model.fit()

        # Define the prediction range
        prediction_start = selected_end + timedelta(days=1)
        prediction_end = prediction_start + timedelta(days=365 * 10)  # Predict for 10 years (1990)

        forecast = model_fit.predict(start=prediction_start, end=prediction_end)

        # Plot the forecasted values
        plt.plot(forecast.index, forecast.values, label="Forecasted")
        plt.legend()
        # Set the x-axis limits for data and forecast
        plt.xlim(selected_start, prediction_end)

        st.pyplot(plt)
    else:
        st.write("Invalid dataset. Please provide a DataFrame or Series.")
