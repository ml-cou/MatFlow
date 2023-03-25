import streamlit as st
import pandas as pd

from modules import utils
from modules.classes import model
from modules.model import build_model, model_report, prediction, delete_model,split_dataset,model_evolution,feature_selection,classification
def model_builder(dataset,table_name):
	try:
		models = st.session_state["models"]
	except:
		st.session_state["models"] = model.Classification()
		models = st.session_state["models"]

	try:
		dataset = st.session_state["dataset"]

	except KeyError:
		st.header("No Dataset Found")
		return

	except Exception as e:
		st.warning(e)
		return

	menus = ["Split Dataset","Model Evolution","Feature Selection","Build Model", "Model Report", "Model Prediction", "Models"]
	tabs = st.tabs(menus)
	print(models)
	#
	with tabs[0]:
		split_dataset.split_dataset(dataset,table_name, models)
	#
	with tabs[1]:
		model_evolution.model_evolution()
	#
	# with tabs[2]:
	# 	prediction.prediction(dataset, models)
	with tabs[3]:
		if 'splitted_data' in st.session_state:
			classification.classification(dataset,models)
		else:
			st.header('Split Dataset First')
	with tabs[4]:
		model_report.model_report(models)
	with tabs[5]:
		prediction.prediction(dataset, models)
	with tabs[6]:
		delete_model.delete_model(models)
