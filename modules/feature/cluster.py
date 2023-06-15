import base64
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import modules.utils as utils

def format_number(num):
    if abs(num) >= 1e6:
        return f"{num/1e6:.0f}M"
    if abs(num) >= 1e3:
        return f"{num/1e3:.0f}K"
    return str(num)

def cluster_dataset(data):
    global fig
    try:
        df = data
        col, col2 = st.columns(2)
        n_cls = col.slider(
            f"Number of classes",
            0, 10, 3,
            key="number_of_classes"
        )
        class_nms = []
        lgn_nms = []
        c = 0
        for i in range(n_cls):
            if c == 0:
                col, col1 = st.columns(2)
            if(i % 2 == 0):
                class_nm = col.text_input(f"Class {i+1} Name", f"Class {i+1}")
            else:
                class_nm = col1.text_input(f"Class {i+1} Name", f"Class {i+1}")
            class_nms.append(class_nm)
            lgn_nms.append(f"{class_nm} ε")
            c = c + 1
            c = c % 2

        col1, col2 = st.columns(2)
        display_type = col1.selectbox(
            "Display Type",
            ["Graph", "Table"]
        )
        col1, col2 = st.columns(2)
        target_col = col1.selectbox(
            f"Target Variable",
            utils.get_variables(data),
            key="model_target"
        )
        feature_cols = [col for col in utils.get_variables(data) if col != target_col]
        if st.button("Submit", key="split_submit_button"):
            X = df[feature_cols].values
            y = df[target_col].values
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            kmeans = KMeans(n_clusters=n_cls, random_state=0).fit(X)
            centroids = pca.transform(kmeans.cluster_centers_)
            fig, ax = plt.subplots()

            # scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_) ## 2k 2 times
            scatter = ax.scatter(y,X_pca[:, 0], c=kmeans.labels_)
            handles, labels = scatter.legend_elements()
            cluster_labels = class_nms
            #centroid_labels = ["" for _ in range(n_cls)]
            ax.legend(handles, lgn_nms)
            # ax.set_title('Extinction Coefficients(ε)')
            # ax.set_xlabel(target_col)
            ax.set_xlabel('Extinction Coefficients(ε)')
            # ax.set_ylabel("Principal Component 2")

            # Format tick labels on the graph
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda num, _: format_number(num)))
            # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda num, _: format_number(num)))
            # formatted_number = "{:.0f}k".format(number / 1000)
            tmp_df=df.copy()
            tmp_df['Category'] = [cluster_labels[label] for label in kmeans.labels_]
            # Calculate the count of each cluster element
            cluster_counts = tmp_df['Category'].value_counts()

            # Create and customize the bar plot
            fig2, ax2 = plt.subplots()
            ax2.bar(cluster_counts.index, cluster_counts.values)
            ax2.set_xlabel('Cluster Category')
            ax2.set_ylabel('Count')
            ax2.set_title('Cluster Element Counts')
            plt.xticks(rotation=45)
            # Add annotations to the bars
            for i, count in enumerate(cluster_counts.values):
                ax2.annotate(format_number(count), xy=(i, count), ha='center', va='bottom')

            if display_type == "Table":
                st.write('#')
                col,col1=st.columns(2)
                col.markdown(download_csv(tmp_df), unsafe_allow_html=True)
                st.write('#')
                st.dataframe(tmp_df)
            else:
                st.pyplot(fig)
                st.write("#")
                st.pyplot(fig2)
    except Exception as e:
        st.error(e)

def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"<a href='data:file/csv;base64,{b64}' download='clustered_dataset.csv'>Download Clustered Dataset</a>"
    return href
