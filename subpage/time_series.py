import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.models import HoverTool
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
from bokeh.plotting import figure

def time_series_analysis(dataset, table_name):
    if 'save_as' not in st.session_state:
        st.session_state.save_as = True

    try:
        data = dataset[table_name]
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
    if(len(date_columns)==0):
        st.warning("No datetime found")
        return
    data.rename(columns={date_columns[0]:"date"},inplace=True)
    data['date'] = pd.to_datetime(data["date"])

    data = data.set_index('date')
    col1,col2=st.columns(2)
    ex_t = col1.number_input(
        f"Extend time",
        key="extended time"
    )

    # Define the time range slider
    min_date_range = data.index.min()
    max_date_range = data.index.max() + timedelta(days=365 * ex_t)

    selected_range = st.slider("Select a time range:",
                               min_value=min_date_range,
                               max_value=max_date_range,
                               value=(data.index.max().to_pydatetime()))

    # Convert selected range to Timestamp objects
    selected_start = min_date_range
    selected_end = selected_range

    # Filter the data based on the selected range
    filtered_data = data.loc[selected_start:selected_end]

    col1, col2 = st.columns(2)
    column = col1.selectbox(
        f"select column",
        filtered_data.columns,
        key="select column"
    )
    # Plot the time series
    st.write("#")
    st.write("Time Series Plot:")
    if isinstance(filtered_data, pd.DataFrame):
        p = figure(
            title=f"Time Series Plot - {column}",
            x_axis_label="Time",
            y_axis_label="Value",
            width=800,
            height=400
        )

        # Add the line plot
        p.line(filtered_data.index, filtered_data[column], line_width=2)

        # Perform forecasting
        model = ARIMA(filtered_data[column], order=(1, 1, 1))
        model_fit = model.fit()

        # Define the prediction range
        prediction_start = selected_start
        prediction_end = selected_end
        forecast = model_fit.predict(start=prediction_start, end=prediction_end)

        # Add the forecasted values
        p.line(forecast.index, forecast.values, line_width=2, color='orange', legend_label="Forecasted")
        # Add the forecasted line plot
        forecast_line = p.line(forecast.index, forecast.values, line_width=2, color='orange', legend_label="Forecasted")

        # Configure hover tool to display values
        hover = HoverTool(renderers=[forecast_line],tooltips=[("Value", "@y")], mode="vline")
        p.add_tools(hover)
        # Adjust legend position
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        st.bokeh_chart(p, use_container_width=True)

    elif isinstance(filtered_data, pd.Series):
        p = figure(
            title="Time Series Plot",
            x_axis_label="Time",
            y_axis_label="Value",
            width=800,
            height=400
        )
        # Add the line plot
        p.line(filtered_data.index, filtered_data.values, line_width=2, legend_label="Data")

        # Perform forecasting
        model = ARIMA(filtered_data, order=(1, 1, 1))
        model_fit = model.fit()

        # Define the prediction range
        prediction_start = selected_start
        prediction_end = selected_end

        forecast = model_fit.predict(start=prediction_start, end=prediction_end)

        # Add the forecasted values
        p.line(forecast.index, forecast.values, line_width=2, color='orange', legend_label="Forecasted")

        # Configure hover tool to display values
        hover = HoverTool(tooltips=[("Value", "@y")], mode="vline")
        p.add_tools(hover)

        # Adjust legend position
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        # Create a column layout with the plot and legend
        plot_layout = column(p)

        st.bokeh_chart(plot_layout, use_container_width=True)

    else:
        st.write("Invalid dataset. Please provide a DataFrame or Series.")
