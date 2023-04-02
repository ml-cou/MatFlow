import streamlit as st
import pandas as pd
def model_deployment():
    try:
        model=st.session_state["models"]
        train_data_name=st.session_state.splitted_data['train_name']
        train_data=st.session_state.dataset.get_data(train_data_name)
        target_var=st.session_state.splitted_data['target_var']
    except:
        st.header('Create a Model First');
        return
    col_names=[]
    for i in train_data.columns:
        if i==target_var:
            continue
        col_names.append(i)
    col1,col2,col3,col4=st.columns([4,0.5,2,0.5])
    with col1:
        model_name = st.selectbox('Select Model', model.list_name())
    col1,col2,col3,col4=st.columns([4,0.5,2,0.5])
    prediction=['']
    with col1:
        st.header('INPUT')
        for i in col_names:
            st.number_input(i,key=i)
        st.write('#')
        if st.button('Submit',type="primary"):
            X=[]
            for i in col_names:
                X.append(st.session_state[i])

            prediction=model.get_prediction(model_name,[X])
    with col3:
        st.header('PREDICTION')
        st.header('#')
        st.write('#')
        st.write('#')
        st.write('#')
        st.markdown(f'<h4> { target_var.upper() } : { prediction[0] }</h4>',
                    unsafe_allow_html=True
                    )






