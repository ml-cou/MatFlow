# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.cluster import KMeans
# # from sklearn.decomposition import PCA
# # import matplotlib.pyplot as plt
# # from IPython.display import clear_output
# # # #
# # def cluster(dataset):
# #     st.set_option('deprecation.showPyplotGlobalUse', False)
# #
# #     st.title('K-Means Clustering Algorithm')
# #     #
# #     # # Load data
# #     # @st.cache
# #     # def load_data():
# #     #     data = pd.read_csv('data.csv')
# #     #     return data
# #
# #     data = dataset
# #
# #     # Display dataset
# #     st.subheader('Dataset')
# #     # st.write(data)
# #
# #     # Data preprocessing
# #     X = data.iloc[:, [3, 4]].values
# #
# #     # Determine optimal number of clusters using elbow method
# #     wcss = []
# #     for i in range(1, 11):
# #         kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
# #         kmeans.fit(X)
# #         wcss.append(kmeans.inertia_)
# #
# #     fig1, ax1 = plt.subplots()
# #     ax1.plot(range(1, 11), wcss)
# #     ax1.set_title('Elbow Method')
# #     ax1.set_xlabel('Number of clusters')
# #     ax1.set_ylabel('WCSS')
# #     st.pyplot(fig1)
# #
# #     # Build model
# #     num_clusters = st.slider('Select the number of clusters:', min_value=1, max_value=10)
# #     kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
# #     y_kmeans = kmeans.fit_predict(X)
# #
# #     # Visualize clusters
# #     fig2, ax2 = plt.subplots()
# #     ax2.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
# #     ax2.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
# #     ax2.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
# #     ax2.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
# #     ax2.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# #     ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
# #     ax2.set_title('Clusters')
# #     ax2.set_xlabel('X Label')
# #     ax2.set_ylabel('Y Label')
# #     ax2.legend()
# #     # st.pyplot(fig2)
# #
# #     fig = plt.figure(figsize=(10, 5))
# #     ax1 = plt.subplot(121)
# #     ax2 = plt.subplot(122)
# # #
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA
# from IPython.display import clear_output
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
#
# def random_centroids(data, k):
#     centroids = []
#     for i in range(k):
#         centroid = data.apply(lambda x: float(x.sample()))
#         centroids.append(centroid)
#     return pd.concat(centroids, axis=1)
#
# def get_labels(data, centroids):
#     distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
#     return distances.idxmin(axis=1)
# def new_centroids(data, labels, k):
#     centroids = data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
#     return centroids
# def plot_clusters(data, labels, centroids, iteration):
#     pca = PCA(n_components=2)
#     data_2d = pca.fit_transform(data)
#     centroids_2d = pca.transform(centroids.T)
#     clear_output(wait=True)
#     plt.title(f'Iteration {iteration}')
#     plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
#     plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
#     plt.show()
#
# def cluster(data):
#     centroids = random_centroids(data, 5)
#     labels = get_labels(data, centroids)
#     labels.value_counts()
#     max_iterations = 100
#     centroid_count = 3
#
#     centroids = random_centroids(data, centroid_count)
#     old_centroids = pd.DataFrame()
#     iteration = 1
#
#     while iteration < max_iterations and not centroids.equals(old_centroids):
#         old_centroids = centroids
#
#         labels = get_labels(data, centroids)
#         centroids = new_centroids(data, labels, centroid_count)
#         plot_clusters(data, labels, centroids, iteration)
#         iteration += 1
#     kmeans = KMeans(3)
#     kmeans.fit(data)
#     # pd.DataFrame(kmeans.cluster_centers_, columns=features)
#     # boston=dataset
#
#     # # standardize the data
#     # scaler = StandardScaler()
#     # X = scaler.fit_transform(boston.data)
#     #
#     # # perform k-means clustering
#     # kmeans = KMeans(n_clusters=3, random_state=42)
#     # y_kmeans = kmeans.fit_predict(X)
#     #
#     # # plot the clusters
#     # plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
#     # centers = kmeans.cluster_centers_
#     # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     # plt.show()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from IPython.display import clear_output
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from modules import utils

def random_centroids(data, k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)


def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)


def new_centroids(data, labels, k):
    centroids = data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids


def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    fig, ax = plt.subplots()
    ax.set_title(f'Iteration {iteration}')
    ax.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels)
    ax.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1])
    st.pyplot(fig)
    plt.clf()
    # clear_output(wait=True)

def cluster_dataset(data):
            #----------only fig-------------------------------
    # df = data
    # col,col2=st.columns(2)
    # n_cls= col.slider(
    #     f"Number of cluster",
    #     0, 10, 3,
    #     key="number of cluster"
    # )
    # # st.write(data)
    # # Extract the features from the dataframe
    # X = df.iloc[:, :-1].values
    #
    # # Apply PCA to reduce the dimensionality of the data to 2
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)
    #
    # # Perform K-means clustering on the data
    # kmeans = KMeans(n_clusters=n_cls, random_state=0).fit(X)
    #
    # # Plot the data points colored by their assigned clusters
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_)
    # handles, labels = scatter.legend_elements()
    # ax.legend(handles, [f'Cluster {label}' for label in np.unique(kmeans.labels_)])
    # ax.set_title('K-means Clustering of Dataset')
    # st.pyplot(fig)

    #

    # data = pd.DataFrame.from_dict(data)
    col,col2=st.columns(2)
    n_cls = col.slider(
        f"Number of cluster",
        0, 10, 3,
        key="number of cluster"
        )
    # if st.button("Submit", key="split_submit_button"):
    centroids = random_centroids(data, 5)
    labels = get_labels(data, centroids)
    labels.value_counts()
    max_iterations = 50
    centroid_count = n_cls
    centroids = random_centroids(data, centroid_count)
    old_centroids = pd.DataFrame()
    iteration = 1

    # Create an empty list to store the centroid numbers for each row

    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids
        labels = get_labels(data, centroids)
        centroids = new_centroids(data, labels, centroid_count)

        # Append the centroid numbers to the list
        # centroid_numbers.extend(labels)

        plot_clusters(data, labels, centroids, iteration)
        iteration += 1

        clear_output(wait=True)
    # Check if the number of iterations is greater than the number of rows in the dataset
    # if iteration > len(data):
    # centroid_numbers = centroid_numbers[:len(data)]  # Truncate the list to match the number of rows
    # Add the centroid numbers as a new column to the data DataFrame
    kmeans = KMeans(n_clusters=centroid_count)
    labels = kmeans.fit_predict(data)
    # Add the centroid numbers as a new column to the data DataFrame
    data['Centroid Number'] = labels
    # data['Centroid Number'] = centroid_numbers

    kmeans = KMeans(n_cls)
    kmeans.fit(data)
    features=utils.get_variables(data)
    # st.header("comparison of centronoid")
    # col1,col2=st.columns(2)
    # st.write("#")
    # with col1:
    #     st.write("#")
    #     st.write("Previous centroids")
    #     st.write("#")
    #     st.write(centroids)
    # with col2:
    #     st.write("#")
    #     st.write("centroids after using k means algo ")
    #     st.write("#")
    #     st.write(pd.DataFrame(kmeans.cluster_centers_, columns=features).T)
    st.table(data)



