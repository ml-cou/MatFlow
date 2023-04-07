import streamlit as st
from modules.dataset import read
from modules import utils
from sklearn.model_selection import train_test_split

def split_dataset(dataset,data_opt):
    list_data = dataset.list_name()

    if list_data:
        col1,col2,col3=st.columns(3)
        with col1:
            split_dataset_root_name=st.text_input('Split Dataset Name')


        if  read.validate(split_dataset_root_name, list_data):

            data = dataset.get_data(data_opt)
            with col2:
                target_var = col2.selectbox(
                    "Target Variable",
                    utils.get_variables(data),
                    key="model_target_var"
                )
            with col3:
                if data[target_var].dtype == "float64" or data[target_var].dtype == "int64":
                    type = 'Regressor'
                    st.write('#')
                    st.write(f" ##### {target_var} _is **:blue[Continuous]**._")
                else:
                    type = 'Classification'
                    st.write('#')
                    st.write(f" ##### {target_var} _is **:orange[Categorical]**._ ")

            col1, col2, col3 = st.columns(3)

            train_name = col1.text_input(
                "Train Data Name",
                f"train_{split_dataset_root_name}",
                key="train_data_name"
            )

            test_name = col2.text_input(
                "Test Data Name",
                f"test_{split_dataset_root_name}",
                key="test_data_name"
            )

            variables = utils.get_variables(data, add_hypen=True)
            stratify = col3.selectbox(
                "Stratify",
                variables,
                key="split_stratify"
            )

            col1, col2,col4 = st.columns([3, 4,1])

            test_size = col1.number_input(
                "Test Size",
                0.0, 1.0, 0.5,
                key="split_test_size"
            )

            random_state = col2.slider(
                "Random State",
                0, 1000, 1,
                key="split_random_state"
            )



            col4.markdown("#")
            shuffle = col4.checkbox("Shuffle", True, key="split_shuffle")

            if st.button("Submit", key="split_submit_button"):

                if 'splitted_data' not in st.session_state:
                    st.session_state.splitted_data = {}

                split_list = st.session_state.splitted_data

                is_valid = [read.validate(name, list_data) for name in [train_name, test_name]]

                if all(is_valid):
                    stratify = None if (stratify == "-") else stratify
                    X = data
                    y = data[target_var]
                    X_train, X_test = train_test_split(X, test_size=test_size,random_state=random_state)
                    split_list.update({split_dataset_root_name:{'train_name': train_name, 'test_name': test_name,
                                                      'target_var':target_var,'type':type}})
                    dataset.add(train_name, X_train)
                    dataset.add(test_name, X_test)
                    st.success("Success")
                    utils.rerun()

    else:
        st.header("No Dataset Found!")
