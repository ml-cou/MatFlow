import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def model_report(split_dataset_name, models):

    result_df = pd.DataFrame()
    for i in st.session_state.has_models[split_dataset_name]:
        result_df = pd.concat([result_df, models.get_result(i)], ignore_index=True)

    col1, col2 = st.columns(2)
    display_type = col1.selectbox(
        "Display Type",
        ["Table", "Graph"]
    )

    if display_type == "Table":
        col2.markdown("#")
        include_data = col2.checkbox("Include Data")

        report_table(result_df, include_data)

    else:
        report_graph(result_df, col2)


def report_table(result_df, include_data):
    cols = result_df.columns
    if not include_data:
        cols = [col for col in cols if col not in ["Train Data", "Test Data"]]

    display_result = st.radio(
        "Display Result",
        ["All", "Train", "Test", "Custom"],
        index=2,
        horizontal=True
    )

    if display_result == "Train":
        cols = result_df.columns[result_df.columns.str.contains("Train")].to_list()
        if include_data:
            cols.insert(0, "Model Name")
        else:
            cols[0] = "Model Name"

    elif display_result == "Test":
        cols = result_df.columns[result_df.columns.str.contains("Test")].to_list()
        if include_data:
            cols.insert(0, "Model Name")
        else:
            cols[0] = "Model Name"

    elif display_result == "Custom":
        cols = st.multiselect(
            "Columns",
            cols,
            ["Model Name"]
        )

    st.dataframe(result_df[cols])


def report_graph(result_df, col):
    graph_cols_option = col.radio(
        "Select an option",
        ("All Columns", "Custom Columns")
    )
    if graph_cols_option == "All Columns":
        graph_cols = result_df.columns[3:]
    else:
        graph_cols = col.multiselect(
            "Display Columns",
            result_df.columns[3:],
            default=result_df.columns[3:]
        )
    annot = st.checkbox("Annotate")

    fig, axs = plt.subplots(ncols=len(graph_cols), figsize=(20, 8))
    for i, graph_col in enumerate(graph_cols):
        result_df_sorted = result_df.sort_values(graph_col, ascending=False)
        ax = sns.barplot(data=result_df_sorted, x=graph_col, y="Model Name", ax=axs[i])

        if annot:
            for rect in ax.patches:
                plt.text(1.05 * rect.get_width(),
                         rect.get_y() + 0.5 * rect.get_height(),
                         '%.3f' % float(rect.get_width()),
                         ha='center', va='center'
                         )

        ax.set_title(graph_col)

    st.pyplot(fig)

# annot = st.checkbox("Annotate")
#
# for graph_col in graph_cols:
# 	fig, ax = plt.subplots()
# 	result_df_sorted = result_df.sort_values(graph_col, ascending=False)
# 	ax = sns.barplot(data=result_df_sorted, x=graph_col, y="Model Name")
#
# 	if annot:
# 		for rect in ax.patches:
# 			plt.text(1.05 * rect.get_width(),
# 					 rect.get_y() + 0.5 * rect.get_height(),
# 					 '%.3f' % float(rect.get_width()),
# 					 ha='center', va='center'
# 					 )
#
# 	ax.set_title(graph_col)
# 	st.pyplot(fig)

# graph_cols = col.multiselect(
# 	"Display Columns",
# 	result_df.columns[3:],
# 	default=result_df.columns[3:]
# )
#
# annot = st.checkbox("Annotate")
# for graph_col in graph_cols:
# 	fig, ax = plt.subplots()
# 	result_df_sorted = result_df.sort_values(graph_col, ascending=False)
# 	ax = sns.barplot(data=result_df_sorted, x=graph_col, y="Model Name")
#
# 	if annot:
# 		for rect in ax.patches:
# 			plt.text(1.05 * rect.get_width(),
# 					 rect.get_y() + 0.5 * rect.get_height(),
# 					 '%.3f' % float(rect.get_width()),
# 					 ha='center', va='center'
# 					 )
#
# 	ax.set_title(graph_col)
# 	st.pyplot(fig)
#
