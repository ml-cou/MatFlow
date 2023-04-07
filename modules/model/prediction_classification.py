import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from modules import utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, roc_curve, precision_recall_curve


def prediction(dataset, models,model_opt):

    col1, col2, col3 = st.columns(3)

    data_opt = col1.selectbox(
        "Select Data",
        dataset.list_name()
    )

    target_var = col2.selectbox(
        "Target Variable",
        models.target_var
    )

    if st.checkbox("Show Result"):
        data = dataset.get_data(data_opt)
        X, y = utils.split_xy(data, target_var)
        y_pred = models.get_prediction(model_opt, X)

        col1, col2 = st.columns(2)
        result_opt = col1.selectbox(
            "Result",
            ["Target Value", "Accuracy", "Precision", "Recall", "F1-Score",
             "Classification Report", "Confusion Matrix", "Actual vs. Predicted",
             "Precision-Recall Curve", "ROC Curve"]

        )

        try:
            if y.nunique() > 2:
                multi_average = col2.selectbox(
                    "Multiclass Average",
                    ["micro", "macro", "weighted"],
                    key="prediction_multi_average"
                )
            else:
                multi_average = "binary"

            show_result(y, y_pred, result_opt, multi_average)

        except ValueError as e:
            st.warning(str(e))

        except TypeError as e:
            st.warning(str(e))


def show_result(y, y_pred, result_opt, multi_average):
    le = LabelEncoder()

    if result_opt in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        metric_dict = {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, average=multi_average),
            "Recall": recall_score(y, y_pred, average=multi_average),
            "F1-Score": f1_score(y, y_pred, average=multi_average)
        }

        result = metric_dict.get(result_opt)
        st.metric(result_opt, result)

    elif result_opt == "Target Value":
        result = pd.DataFrame({
            "Actual": y,
            "Predicted": y_pred
        })

        st.dataframe(result)

    elif result_opt == "Classification Report":
        result = classification_report(y, y_pred)

        st.text("`" + result)

    elif result_opt == "Confusion Matrix":
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        ax = sns.heatmap(
            cm, annot=True,
            fmt='.4g',
            xticklabels=np.unique(y_pred),
            yticklabels=np.unique(y_pred)
        )

        ax.set_title(result_opt)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        st.pyplot(fig)

    elif result_opt == "Actual vs. Predicted":
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = sns.lineplot(x=range(len(y)), y=y, label="Actual")
        ax = sns.lineplot(x=range(len(y_pred)), y=y_pred, label="Predicted")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title("Actual vs. Predicted Values")
        st.pyplot(fig)



    elif result_opt == "Precision-Recall Curve":

        if y.nunique() > 2:
            st.warning("Precision-Recall curve is not supported for multiclass classification")

            return

        # encode y and y_pred
        y_encoded = le.fit_transform(y)
        y_pred_encoded = le.transform(y_pred)

        precision, recall, _ = precision_recall_curve(y_encoded.ravel(), y_pred_encoded.ravel())

        fig, ax = plt.subplots()

        sns.lineplot(x=recall, y=precision, ax=ax)

        ax.set_xlabel("Recall")

        ax.set_ylabel("Precision")

        ax.set_title("Precision-Recall Curve")

        st.pyplot(fig)

    elif result_opt == "ROC Curve":

        if y.nunique() > 2:
            st.warning("ROC curve is not supported for multiclass classification")

            return

        y_encoded = le.fit_transform(y)
        y_pred_encoded = le.transform(y_pred)

        fpr, tpr, _ = roc_curve(y_encoded.ravel(), y_pred_encoded.ravel())

        fig, ax = plt.subplots()

        sns.lineplot(x=fpr, y=tpr,ax=ax)

        ax.set_xlabel("False Positive Rate")

        ax.set_ylabel("True Positive Rate")

        ax.set_title("ROC Curve")

        st.pyplot(fig)

    # elif result_opt == "Feature Importances":
    #
    #     if hasattr(models, 'feature_importances_'):
    #
    #         feature_importances = pd.Series(models.feature_importances_, index=X.columns)
    #
    #     elif hasattr(models, 'coef_'):
    #
    #         feature_importances = pd.Series(models.coef_[0], index=X.columns)
    #
    #     else:
    #
    #         st.warning("Model doesn't support feature importances")
    #
    #         return
    #
    #     feature_importances = feature_importances.sort_values(ascending=False)
    #
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #
    #     sns.barplot(x=feature_importances, y=feature_importances.index, ax=ax)
    #
    #     ax.set_xlabel("Feature Importance Score")
    #
    #     ax.set_ylabel("Features")
    #
    #     ax.set_title("Feature Importances")
    #
    #     st.pyplot(fig)
