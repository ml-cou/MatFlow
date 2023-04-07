import streamlit as st
import pickle
from modules.utils import split_xy
from modules.classifier import knn, svm, log_reg, decision_tree, random_forest, perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def classification(dataset, models):
    try:
        train_name = dataset['train_name']
        test_name = dataset['test_name']
        train_data = st.session_state.dataset.get_data(dataset['train_name'])
        test_data = st.session_state.dataset.get_data(dataset['test_name'])
        target_var = dataset["target_var"]
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
    metric_list = ["Accuracy", "Precision", "Recall", "F1-Score"]

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
        metric_list,
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

            if 'all_models' not in st.session_state:
                st.session_state.all_models = {}

            all_models = st.session_state.all_models

            selected_metrics=get_result(model, X_test, y_test, metrics, multi_average)

            for met in metrics:
                st.metric(met, selected_metrics.get(met))

            result = []
            for X, y in zip([X_train, X_test], [y_train, y_test]):
                temp = get_result(model, X, y, metrics, multi_average)
                result += list(temp.values())

            models.add_model(model_name, model, train_name, test_name, target_var, result)
            all_models.update({model_name:('Classification',dataset)})
        else:
            st.warning("Model name already exist!")


def get_result(model, X, y, metrics, multi_average, pos_label=None):
    y_pred = model.predict(X)

    metric_dict = {}

    try:
        pos_label = y[1]
    except:
        pass

    # calculate accuracy
    if "Accuracy" in metrics:
        metric_dict["Accuracy"] = accuracy_score(y, y_pred)

    # calculate precision
    if "Precision" in metrics:
        try:
            metric_dict["Precision"] = precision_score(y, y_pred, average=multi_average, pos_label=pos_label)
        except ValueError:
            metric_dict["Precision"] = "-"

    # calculate recall
    if "Recall" in metrics:
        try:
            metric_dict["Recall"] = recall_score(y, y_pred, average=multi_average, pos_label=pos_label)
        except ValueError:
            metric_dict["Recall"] = "-"

    # calculate F1-score
    if "F1-Score" in metrics:
        try:
            metric_dict["F1-Score"] = f1_score(y, y_pred, average=multi_average, pos_label=pos_label)
        except ValueError:
            metric_dict["F1-Score"] = "-"

    return metric_dict
