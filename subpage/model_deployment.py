import streamlit as st


def model_deployment():
    if 'all_models' not in st.session_state:
        st.header('Create a Model First')
        return

    all_models = st.session_state.all_models

    col1, col2, col3, col4 = st.columns([4, 0.5, 2, 0.5])
    with col1:
        model_name = st.selectbox('Select Model', all_models, key='model_evaluation')

    try:
        dataset = all_models[model_name][1]
    except:
        st.header("Properly Split Dataset First")
        return

    try:
        model = st.session_state["models"]
        train_data_name = dataset['train_name']
        test_data_name=dataset['test_name']
        train_data = st.session_state.dataset.get_data(train_data_name)
        test_data = st.session_state.dataset.get_data(train_data_name)
        target_var = dataset['target_var']
    except:
        st.header('Create a Model First')
        return

    col_names_all = []
    for i in train_data.columns:
        if i == target_var:
            continue
        col_names_all.append(i)

    rad = col3.radio('Select Columns', ['All Columns', 'Custom Columns'])
    if rad == 'All Columns':
        col_names = col_names_all
    else:
        col_names = st.multiselect('Custom Columns', col_names_all, help='Other values will be 0 as default value')

    col1, col2, col3, col4 = st.columns([4, 1, 2, 0.5])
    prediction = ['']
    correlations = train_data[col_names + [target_var]].corr()[target_var]

    with col1:
        st.header('INPUT')
        for i in col_names:
            threshold=train_data[i].abs().max()
            arrow,threshold = ('**:green[↑]**',threshold) if correlations[i] >= 0 else ('**:red[↓]**',-threshold)
            space='&nbsp;'*150
            st.number_input(i + space + str(threshold)+' ' + arrow,value=threshold, key=i)

        st.write('#')
        if st.button('Submit', type="primary"):
            X = [st.session_state[i] if i in col_names else 0 for i in col_names_all]
            prediction = model.get_prediction(model_name, [X])

    # with col2:
    #     st.write('#')
    #     for i in col_names:
    #         threshold = train_data[i].abs().max()
    #         arrow, threshold = ('**:green[↑]**', threshold) if correlations[i] >= 0 else ('**:red[↓]**', -threshold)
    #         st.write(str(threshold)+' '+arrow)
    #         st.write('#')


    with col3:
        st.header('PREDICTION')
        st.header('#')
        st.write('#')
        st.write('#')
        st.write('#')
        st.markdown(f'<h4> {target_var.upper()} : {prediction[0]}</h4>',
                    unsafe_allow_html=True
                    )

