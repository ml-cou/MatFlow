import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from IPython.display import clear_output
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from modules import utils

def cluster_dataset(data):
    try:
        df = data
        col, col2 = st.columns(2)
        n_cls = col.slider(
            f"Number of clusters",
            0, 10, 3,
            key="number_of_clusters"
        )
        X = df.iloc[:, :-1].values
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        kmeans = KMeans(n_clusters=n_cls, random_state=0).fit(X)
        centroids = pca.transform(kmeans.cluster_centers_)


        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_)
        handles, labels = scatter.legend_elements()
        cluster_labels = [f'Cluster {label}' for label in np.unique(kmeans.labels_)]
        centroid_labels = [f'' for i in range(n_cls)]
        ax.legend(handles, cluster_labels + centroid_labels)

        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o')
        for i, txt in enumerate(centroid_labels):
            ax.annotate(txt, (centroids[i, 0], centroids[i, 1]), xytext=(-10, 10),
                        textcoords='offset points', color='red')

        ax.set_title('K-means Clustering of Dataset')

        kmeans = KMeans(n_cls)
        labels = kmeans.fit_predict(data)
        # Add the centroid numbers as a new column to the data DataFrame
        data['Class'] = labels
        kmeans = KMeans(n_cls)
        kmeans.fit(data)

        col1, col2 = st.columns(2)
        display_type = col1.selectbox(
            "Display Type",
            ["Table", "Graph"]
        )

        if display_type == "Table":
            col2.markdown("#")
            st.table(data)
        else:
            st.pyplot(fig)
    except Exception as e:
        st.error(e)
