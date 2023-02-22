import streamlit as st 
import pandas as pd

from modules import utils  
from modules.classes import model
from modules.model import build_model, model_report, prediction, delete_model
def ds_final_data_analysis(ds):
	try:
		models = st.session_state["models"]

	except:
		st.session_state["models"] = model.Classification()
		models = st.session_state["models"]

	try:
		dataset = ds.file_data

	except KeyError:
		st.header("No Dataset Found")
		st.stop()

	except Exception as e:
		st.warning(e)
		st.stop()

	menus = ["Build Model", "Model Report", "Model Prediction", "Delete Model"]
	tabs = st.tabs(menus)

	with tabs[0]:
		build_model.build_model(dataset, models)

	with tabs[1]:
		model_report.model_report(models)

	with tabs[2]:
		prediction.prediction(dataset, models)

	with tabs[3]:
		delete_model.delete_model(models)
