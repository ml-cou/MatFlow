import streamlit as st


def change_field_name(file_):
    fixed_li = file_.file_data.columns.values.tolist()
    col1,col2,col3=st.columns(3)
    with col1:
        n_iter = st.number_input(
            "Number of Columns",
            1, len(fixed_li), 1,
            key="change_fname_n_rows"
        )
    col1, col2 = st.columns([5, 5])
    selected = []

    for i in range(n_iter):
        with col1:
            var = st.selectbox(
                "Field name",
                options=fixed_li,
                key=f"change_fname_var_{i}"
            )
            selected.append(var)
            fixed_li = [key for key in fixed_li if key not in selected]
        with col2:
            var2 = st.text_input('New field name',
                key=f"change_fname_rename_{i}"
            )
        st.session_state.alt_val.__setitem__(var,var2)