import streamlit as st
import pandas as pd

from modules import utils
from modules.classes import model
from modules.model import build_model, model_report, prediction, delete_model,split_dataset,model_evaluation,feature_selection,classification,regression
def model_builder(dataset,table_name):

	try:
		dataset = st.session_state["dataset"]

	except KeyError:
		st.header("No Dataset Found")
		return

	except Exception as e:
		st.warning(e)
		return

	menus = ["Split Dataset","Build Model", "Model Report", "Model Prediction","Model Evaluation", "Models"]
	tabs = st.tabs(menus)
	#

	with tabs[0]:
		split_dataset.split_dataset(dataset,table_name)
	#
	with tabs[4]:
		model_evaluation.model_evaluation()
	with tabs[1]:
		if 'splitted_data' in st.session_state:
			if st.session_state.splitted_data['type']=='Classifier':
				try:
					models = st.session_state["models"]
				except:
					st.session_state["models"] = model.Classification()
					models = st.session_state["models"]
				classification.classification(dataset,models)
			else:
				try:
					models = st.session_state["models"]
				except:
					st.session_state["models"] = model.Regressor()
					models = st.session_state["models"]
				regression.regression(dataset,models)
		else:
			st.header('Split Dataset First')
	with tabs[2]:
		if "models" in st.session_state:
			model_report.model_report(models)
		else:
			st.header('No Model Found')
	with tabs[3]:
		if "models" in st.session_state:
			prediction.prediction(dataset, models)
		else:
			st.header('No Model Found')
	with tabs[5]:
		if "models" in st.session_state:
			delete_model.delete_model(models)
		else:
			st.header('No Model Found')
