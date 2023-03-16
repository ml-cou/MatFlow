import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from modules import utils
from modules.dataset import read
from modules.model import classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from modules.classifier import decision_tree,knn,log_reg,random_forest,svm,perceptron


def build_model(dataset,data_opt, models):
#---------------split dataset----------------------
	list_data = dataset.list_name()
	if list_data:
		col1, col2, col3 = st.columns(3)
	# 	data_opt = col1.selectbox(
	# 			"Dataset",
	# 			list_data,
	# 			key="dataset_split_options"
	# 		)
		train_name = col2.text_input(
				"Train Data Name",
				f"{data_opt}_train",
				key="train_data_name"
			)

		test_name = col3.text_input(
				"Test Data Name",
				f"{data_opt}_test",
				key="test_data_name"
			)

		data = dataset.get_data(data_opt)
		variables = utils.get_variables(data, add_hypen=True)

		col1, col2, col3, col4 = st.columns([2,4,2,2])
		test_size = col1.number_input(
				"Test Size",
				0.1, 1.0, 0.2,
				key="split_test_size"
			)

		stratify = col2.selectbox(
				"Stratify",
				variables,
				key="split_stratify"
			)

		random_state = col3.number_input(
				"Random State",
				0, 1000, 0,
				key="split_random_state"
			)

		col4.markdown("#")
		shuffle = col4.checkbox("Shuffle", True, key="split_shuffle")

		if st.button("Submit", key="split_submit_button"):
			is_valid = [read.validate(name, list_data) for name in [train_name, test_name]]

			if all(is_valid):
				stratify = None if (stratify == "-") else stratify
				df_train, df_test = train_test_split(data, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)

				df_train.reset_index(drop=True, inplace=True)
				df_test.reset_index(drop=True, inplace=True)

				dataset.add(train_name, df_train)
				dataset.add(test_name, df_test)

				st.success("Success")
				utils.rerun()

	else:
		st.header("No Dataset Found!")

#-----------------spilt dataset end---------------------

	data_opt = dataset.list_name()
	default_idx = 0

	col1, col2, col3 = st.columns(3)
	train_name = col1.selectbox(
		"Train Data",
		data_opt,
		default_idx,
		key="model_train_data"
	)

	test_name = col2.selectbox(
		"Test Data",
		data_opt,
		default_idx,
		key="model_test_data"
	)

	train_data = dataset.get_data(train_name)

	target_var = col3.selectbox(
		"Target Variable",
		utils.get_variables(train_data),
		key="model_target_var"
	)
	# 	classification.classification(dataset, models, train_name, test_name, target_var)

	# Load data
	data = train_data
	st.write("ML type: ")
	col1, col2 = st.columns([4,4])
	options = ["Regresion", "Classification"]

	if data[target_var].dtype == "float64" or data[target_var].dtype == "int64":
		st.write(f"{target_var} is numerical")
		default_option = "Regresion"
		with col1:
			selected_option = st.radio("", options, index=options.index(default_option))

		#--------Regresion -----------------

		X = data
		y = data[target_var]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

		reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
		models, predictions = reg.fit(X_train, X_test, y_train, y_test)
		with col1:
			st.table(models)

	else:
		default_option = "Classification"
		with col1:
			selected_option = st.radio("", options, index=options.index(default_option))
			st.write(f"{target_var} is not numerical")

		### separating dataset into dependent and independent features
		X = data
		y = data[target_var]
		### splitting dataset into training and testing part(50% training and 50% testing)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size , random_state=random_state )

		clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
		models, predictions = clf.fit(X_train, X_test, y_train, y_test)

		results_df = models

		# Create a selectbox to choose a column from the dataframe
		# selected_column = st.selectbox('Select a column', options=results_df.columns)

		# Get the values of the selected column
		# selected_column_values = results_df[selected_column].tolist()

		# Display the values of the selected column
		# st.write(selected_column_values)
		idx=models.head()
		# st.write(idx.index.values)
		listCls=[]
		# with col1:
		st.write(models.index)
		for i in models.index:
			if(i=="decision_tree"):
				listCls.insert(i,decision_tree())
			elif (i=="knn"):
				listCls.insert(i,knn())
			elif (i=="log_reg"):
				listCls.insert(i,log_reg())
			elif (i=="perceptron"):
				listCls.insert(i,perceptron())
			elif (i=="random_forest"):
				listCls.insert(i,random_forest())
			elif (i=="svm"):
				listCls.insert(i,svm())




		models.insert(loc=5, column="Model name", value=[1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4])
		st.write(models)
		# svm.svm()




