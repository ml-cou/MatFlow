import streamlit as st  

from sklearn.neighbors import KNeighborsClassifier

def knn():
	col1, col2, col3 = st.columns(3)
	n_neighbors = col1.number_input(
			"Number of Neighbors",
			1,100,3,   
			key="knn_neighbors"
		)
	weights = col2.selectbox(
			"Weight Function",
			["uniform", "distance"],
			key="knn_weight"
		)
	metric = col3.selectbox(
			"Distance Metric",
			["minkowski", "euclidean", "manhattan"]
		)

	try:
		model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
	except ValueError as e:
		print("An error occurred while creating the KNeighborsClassifier model: ", e)
	except Exception as e:
		print("An unexpected error occurred: ", e)

	return model