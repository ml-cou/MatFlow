import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from modules import utils


def comparison_plot(data, table_name):
    num_var = utils.get_numerical(data)

    try:
        selected_features = st.session_state.selected_feature[table_name]
    except:
        selected_features = []

    col1, col2, col3 = st.columns([4, 4, 2])
    x_var = col1.multiselect(
        "X Variable",
        ['-', 'selected_features'] + num_var if len(selected_features) > 0 else num_var,
        key="custom_x_var"
    )

    y_var = col2.selectbox(
        "Y Variable",
        num_var,
        key="custom_y_var"
    )

    hue_var = col3.selectbox(
        "Hue Variable",
        ['None'] + utils.get_low_cardinality(data),
        key="custom_hue_var"
    )


    if 'selected_features' in x_var:
        x_var.remove('selected_features')
        x_var.extend(selected_features)
        x_var = list(set(x_var))

    feature_dict = {}

    for i in num_var:
        feature_dict.update({i: i})

    with st.expander('Rename Features'):
        for feature in x_var:
            renamed_feature = st.text_input(f"Rename '{feature}' to:", feature)
            feature_dict[feature] = renamed_feature

    graph_type = st.multiselect('Select Graph Type', ['Line Plot', 'Scatter Plot'], default=["Line Plot"])


    colors = sns.color_palette('husl', (len(x_var) + 1) * (1 if hue_var == 'None' else len(data[hue_var].unique())))
    try:
        legend_colors = {}
        labels = []
        tmp_labels = []
        for i, feature in enumerate(x_var):
            tmp = []
            if hue_var == 'None':
                legend_colors.update({feature_dict.get(feature, feature): colors[i]})
                tmp.append(feature)
            else:
                hue_labels = data[hue_var].unique()
                hue_labels.sort()
                for j, label in enumerate(hue_labels):
                    label_suffix = f"{feature_dict.get(feature, feature)} {label}"
                    legend_colors.update({label_suffix: colors[i * len(x_var) + j]})
                    tmp.append(label_suffix)
            tmp_labels.append(tmp)

        for i in range(1 if hue_var == 'None' else len(data[hue_var].unique())):
            for j in range(len(x_var)):
                labels.append(tmp_labels[j][i])
    except Exception as e:
        pass

    try:
        if len(graph_type) > 1:
            fig, axs = plt.subplots(1, 2, figsize=(20, max(min(len(x_var), 15), 9)))
            line_plot(x_var, hue_var, data, y_var, feature_dict, legend_colors, axs[0], labels)
            scatter_plot(x_var, hue_var, data, y_var, feature_dict, legend_colors, axs[1], labels, False)
        elif 'Line' in graph_type[0]:
            fig, axs = plt.subplots(1, 1, figsize=(20, max(min(len(x_var), 15), 9)))
            line_plot(x_var, hue_var, data, y_var, feature_dict, legend_colors, axs, labels)
        else:
            fig, axs = plt.subplots(1, 1, figsize=(20, max(min(len(x_var), 15), 9)))
            scatter_plot(x_var, hue_var, data, y_var, feature_dict, legend_colors, axs, labels, True)

        st.pyplot(plt)
    except Exception as e:
        st.error("An error occurred while plotting the data.")
        st.warning(str(e))


def line_plot(x_var, hue_var, data, y_var, feature_dict, legend_colors, axs, labels):
    for i, feature in enumerate(x_var):
        if hue_var == 'None':
            sns.lineplot(data=data, x=feature, y=y_var, label=feature_dict.get(feature, feature),
                         color=legend_colors[feature], ax=axs, linewidth=2.5)
        else:
            hue_labels = data[hue_var].unique()
            hue_labels.sort()
            for j, label in enumerate(hue_labels):
                hue_data = data[data[hue_var] == label]
                label_suffix = f"{feature_dict.get(feature, feature)} {label}"
                sns.lineplot(data=hue_data, x=feature, y=y_var, label=label_suffix,
                             color=legend_colors[label_suffix], ax=axs, linewidth=2.5)

    axs.set_xlabel('Features')
    axs.set_ylabel(feature_dict[y_var])

    axs.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', labels=labels,
                  ncol=1 if hue_var == 'None' else len(data[hue_var].unique()))
    for legend_handle, feature_label in zip(axs.get_legend().legendHandles, labels):
        legend_handle.set_color(legend_colors.get(feature_label, 'black'))
    plt.tight_layout()


def scatter_plot(x_var, hue_var, data, y_var, feature_dict, legend_colors, axs, labels, show_legend):
    # Scatter plot for combined data
    for i, feature in enumerate(x_var):
        if hue_var == 'None':
            sns.scatterplot(data=data, x=feature, y=y_var, label=feature_dict.get(feature, feature),
                            color=legend_colors[feature], ax=axs)
        else:
            hue_labels = data[hue_var].unique()
            hue_labels.sort()
            for j, label in enumerate(hue_labels):
                hue_data = data[data[hue_var] == label]
                label_suffix = f"{feature_dict.get(feature, feature)} {label}"
                sns.scatterplot(data=hue_data, x=feature, y=y_var, label=label_suffix,
                                color=legend_colors[label_suffix], ax=axs)

    axs.set_xlabel('Features')
    axs.set_ylabel(feature_dict[y_var])
    if show_legend:
        axs.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', labels=labels,
                      ncol=1 if hue_var == 'None' else len(data[hue_var].unique()))
    else:
        axs.legend().set_visible(False)

    plt.tight_layout()
    
