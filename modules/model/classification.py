import streamlit as st
import pickle
from modules.utils import split_xy
from modules.classifier import knn, svm, log_reg, decision_tree, random_forest, perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def classification(dataset, models):
	try:
		train_name=st.session_state.splitted_data['train_name']
		test_name=st.session_state.splitted_data['test_name']
		train_data = st.session_state.dataset.get_data(st.session_state.splitted_data['train_name'])
		test_data = st.session_state.dataset.get_data(st.session_state.splitted_data['test_name'])
		target_var=st.session_state.splitted_data["target_var"]
		X_train, y_train = split_xy(train_data, target_var)
		X_test, y_test = split_xy(test_data, target_var)
	except:
		st.header("Properly Split Dataset First")
		return

	try:
		X_train, X_test = X_train.drop(target_var, axis=1), X_test.drop(target_var, axis=1)
	except:
		pass

	classifiers = {
		"K-Nearest Neighbors": "KNN", 
		"Support Vector Machine": "SVM", 
		"Logistic Regression": "LR",
		"Decision Tree": "DT",
		"Random Forest": "RF",
		"Multilayer Perceptron": "MLP"
	}
	metric_list = ["Accuracy", "Precision","Recall", "F1-Score"]

	st.markdown("#")
	col1, col2 = st.columns([6.66, 3.33])
	classifier = col1.selectbox(
			"Classifier",
			classifiers.keys(),
			key="model_classifier"
		)

	model_name = col2.text_input(
			"Model Name",
			classifiers[classifier],
			key="model_name"
		)

	if classifier == "K-Nearest Neighbors":
		model = knn.knn()
	elif classifier == "Support Vector Machine":
		model = svm.svm()
	elif classifier == "Logistic Regression":
		model = log_reg.log_reg()
	elif classifier == "Decision Tree":
		model = decision_tree.decision_tree()
	elif classifier == "Random Forest":
		model = random_forest.random_forest()
	elif classifier == "Multilayer Perceptron":
		model = perceptron.perceptron()

	col1, col2 = st.columns([6.66, 3.33])
	metrics = col1.multiselect(
			"Display Metrics",
			metric_list,
			["Accuracy"],
			key="model_clf_metrics"
		)

	target_nunique = y_train.nunique()
	if target_nunique > 2:
		multi_average = col2.selectbox(
				"Multiclass Average",
				["micro", "macro", "weighted"],
				key="clf_multi_average"
			)
	else:
		multi_average = "binary"


	if st.button("Submit", key="model_submit"):
		if model_name not in models.list_name():
			try:
				model.fit(X_train, y_train)
			except Exception as e:
				st.error(e)
				return

			selected_metrics = get_result(model, X_test, y_test, metrics, multi_average)
			for met in metrics:
				st.metric(met, selected_metrics.get(met))

			result = []
			for X, y in zip([X_train, X_test], [y_train, y_test]):
				temp = get_result(model, X, y, metrics, multi_average)
				result += list(temp.values())

			models.add_model(model_name, model, train_name, test_name, target_var, result)
		else:
			st.warning("Model name already exist!")



def get_result(model, X, y, metrics, multi_average):
	y_pred = model.predict(X)
	metric_dict = {
		"Accuracy": accuracy_score(y, y_pred),
		"Precision": precision_score(y, y_pred, average=multi_average),
		"Recall": recall_score(y, y_pred, average=multi_average),
		"F1-Score": f1_score(y, y_pred, average=multi_average)
	}

	# only get selected metrics and ignore the others
	# result = {metric: metric_dict.get(metric) for metric in metrics}

	# get selected metrics and fill the others with "-"
	# result = {key: res if key in metrics else "-" for key, res in metric_dict.items()}

	result = metric_dict

	return result
