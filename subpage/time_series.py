import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


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

    # Display the dataset
    st.write("Original Dataset:")
    st.write(data)
    data['date'] = pd.to_datetime(data['date'])

    data = data.set_index('date')
    # Plot the time series
    st.write("Time Series Plot:")
    if isinstance(data, pd.DataFrame):
        for column in data.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(data[column])
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.title(f"Time Series Plot - {column}")
            st.pyplot(plt)
    elif isinstance(data, pd.Series):
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Time Series Plot")
        st.pyplot(plt)
    else:
        st.write("Invalid dataset. Please provide a DataFrame or Series.")

    # Perform seasonal decomposition
    st.write("Seasonal Decomposition:")
    if isinstance(data, pd.DataFrame):
        for column in data.columns:
            decomposition = seasonal_decompose(data[column], model='additive', period=12)
            fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))
            decomposition.observed.plot(ax=ax[0], legend=False)
            ax[0].set_ylabel('Observed')
            decomposition.trend.plot(ax=ax[1], legend=False)
            ax[1].set_ylabel('Trend')
            decomposition.seasonal.plot(ax=ax[2], legend=False)
            ax[2].set_ylabel('Seasonal')
            decomposition.resid.plot(ax=ax[3], legend=False)
            ax[3].set_ylabel('Residuals')
            plt.tight_layout()
            st.pyplot(plt)
    elif isinstance(data, pd.Series):
        decomposition = seasonal_decompose(data, model='additive', period=12)
        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))
        decomposition.observed.plot(ax=ax[0], legend=False)
        ax[0].set_ylabel('Observed')
        decomposition.trend.plot(ax=ax[1], legend=False)
        ax[1].set_ylabel('Trend')
        decomposition.seasonal.plot(ax=ax[2], legend=False)
        ax[2].set_ylabel('Seasonal')
        decomposition.resid.plot(ax=ax[3], legend=False)
        ax[3].set_ylabel('Residuals')
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.write("Invalid dataset. Please provide a DataFrame or Series.")

    # Perform Augmented Dickey-Fuller test
    st.write("Augmented Dickey-Fuller Test:")
    if isinstance(data, pd.DataFrame):
        for column in data.columns:
            result = adfuller(data[column].squeeze())
            st.write("Column:", column)
            st.write("ADF Statistic:", result[0])
            st.write("p-value:", result[1])
            st.write("Critical Values:")
            for key, value in result[4].items():
                st.write(f"{key}: {value}")
    elif isinstance(data, pd.Series):
        result = adfuller(data.squeeze())
        st.write("ADF Statistic:", result[0])
        st.write("p-value:", result[1])
        st.write("Critical Values:")
        for key, value in result[4].items():
            st.write(f"{key}: {value}")
    else:
        st.write("Invalid dataset. Please provide a DataFrame or Series.")

# def time_series_analysis(dataset,table_name):
#
#     if 'save_as' not in st.session_state:
#         st.session_state.save_as = True
#     try:
#
#         data = dataset[table_name]
#         data_opt = table_name
#     except KeyError:
#         st.header("No Dataset Found")
#         st.stop()
#
#     except Exception as e:
#         st.warning(e)
#         st.stop()
#
#     st.subheader("Time Series Analysis")
#
#     # Display the dataset
#     st.write("Original Dataset:")
#     st.write(data)
#
#     # # Plot the time series
#     # st.write("Time Series Plot:")
#     # plt.figure(figsize=(10, 6))
#     # plt.plot(data)
#     # plt.xlabel("Time")
#     # plt.ylabel("Value")
#     # plt.title("Time Series Plot")
#     # st.pyplot(plt)
#     #
#     # # Perform seasonal decomposition
#     # st.write("Seasonal Decomposition:")
#     # decomposition = seasonal_decompose(data, model='additive', period=12)  # Adjust the period based on your dataset
#     # fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))
#     # decomposition.observed.plot(ax=ax[0], legend=False)
#     # ax[0].set_ylabel('Observed')
#     # decomposition.trend.plot(ax=ax[1], legend=False)
#     # ax[1].set_ylabel('Trend')
#     # decomposition.seasonal.plot(ax=ax[2], legend=False)
#     # ax[2].set_ylabel('Seasonal')
#     # decomposition.resid.plot(ax=ax[3], legend=False)
#     # ax[3].set_ylabel('Residuals')
#     # plt.tight_layout()
#     # st.pyplot(plt)
#     #
#     # # Perform Augmented Dickey-Fuller test
#     # st.write("Augmented Dickey-Fuller Test:")
#     # result = adfuller(data.squeeze())
#     # st.write("ADF Statistic:", result[0])
#     # st.write("p-value:", result[1])
#     # st.write("Critical Values:")
#     # for key, value in result[4].items():
#     #     st.write(f"{key}: {value}")
#     #
#     #
#
#
#     #-------multivariatee-------
#     st.write(type(data))
#     if isinstance(data, pd.Series):
#         data=pd.Series(data)
#         data = data.to_frame()
#     fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10, 6))
#     for i, ax in enumerate(axes.flatten()):
#         data = data[data.columns[i]]
#         ax.plot(data, color='red', linewidth=1)
#         # Decorations
#         ax.set_title(data.columns[i])
#         ax.xaxis.set_ticks_position('none')
#         ax.yaxis.set_ticks_position('none')
#         ax.spines["top"].set_alpha(0)
#         ax.tick_params(labelsize=6)
#
#     plt.tight_layout()

    # # Load the Air Passengers dataset
    # # data = pd.read_csv('air_passengers.csv')
    # data['Month'] = pd.to_datetime(data['Month'])
    # data.set_index('Month', inplace=True)
    #
    # # Visualize the time series data
    # plt.figure(figsize=(10, 6))
    # plt.plot(data)
    # plt.xlabel('Year')
    # plt.ylabel('Passenger Count')
    # plt.title('Air Passengers Dataset')
    # plt.show()
    #
    # # Perform seasonal decomposition
    # decomposition = seasonal_decompose(data, model='additive')
    # trend = decomposition.trend
    # seasonal = decomposition.seasonal
    # residual = decomposition.resid
    #
    # # Visualize the decomposed components
    # plt.figure(figsize=(10, 10))
    # plt.subplot(411)
    # plt.plot(data, label='Original')
    # plt.legend(loc='best')
    # plt.subplot(412)
    # plt.plot(trend, label='Trend')
    # plt.legend(loc='best')
    # plt.subplot(413)
    # plt.plot(seasonal, label='Seasonality')
    # plt.legend(loc='best')
    # plt.subplot(414)
    # plt.plot(residual, label='Residuals')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()
    #
    # # Perform Augmented Dickey-Fuller test for stationarity
    # result = adfuller(data['#Passengers'])
    # print('ADF Statistic:', result[0])
    # print('p-value:', result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t', key, ':', value)
