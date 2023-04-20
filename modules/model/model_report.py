import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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




def report_graph(data, col):
    model_data = data.drop(columns=['Train Data', 'Test Data', 'Model Name'])

    # Create the figure and axis objects
    fig, ax = plt.subplots(nrows=1, ncols=len(model_data.columns), figsize=(16,0.65*(len(model_data.columns))))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # Define the color map
    cmap = plt.cm.get_cmap('Set3', len(model_data))

    for i, col in enumerate(model_data.columns):
        # Loop over the rows and plot the bars with different colors for each row
        for j, row in enumerate(model_data.iterrows()):
            ax[i].bar(j, row[1][i], color=cmap(j), label=list(data['Model Name'].values)[j])
            ax[i].set_xlabel(col)
            ax[i].set_xticklabels([])
    ax[0].set_ylabel("Value")
    ax[-1].legend(loc='upper left', bbox_to_anchor=(0, 1.5))
    # Add some padding to the top of the plot to make space for the legend
    fig.subplots_adjust(top=0.85 + 0.05 * len(data))
    # Add some padding to the plot
    plt.tight_layout()
    st.pyplot(fig)


    ### rotation

    # for i, col in enumerate(model_data.columns):
    #     # Loop over the rows and plot the bars with different colors for each row
    #     for j, row in enumerate(model_data.iterrows()):
    #         ax[i].barh(j, row[1][i], color=cmap(j), label=list(data['Model Name'].values)[j])
    #         ax[i].set_xlabel(col)
    # ax[0].set_ylabel("Value")
    # ax[-1].legend(loc='upper left', bbox_to_anchor=(0, 0.35 * len(data) + 1))
    # # Add some padding to the top of the plot to make space for the legend
    # fig.subplots_adjust(top=0.85 + 0.05 * len(data))
    # # Add some padding to the plot
    # plt.tight_layout()
    #
    # # Display the plot in the Streamlit app
    # st.pyplot(fig)
