import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

menu = ["MatFlow", "Demo", 'Help', "Start"]


def vspace(n):
    for i in range(n):
        st.markdown('<br>', unsafe_allow_html=True)


def navbar():

    st.markdown('''
    <style>
    .main{
    background:rgb(227 231 238);
    }
    [data-testid="stHeader"]
    {
    display:none;
    }
    .css-k1ih3n
    {
    padding:0px !important;
    margin:0px !important;
    }
    [data-testid="stVerticalBlock"]
    {
    gap:0px;
    margin-bottom:30px !important;
    }
    </style>
    ''', unsafe_allow_html=True)

    if 'page_index' not in st.session_state:
        st.session_state.page_index = 0
        switch_page('home')

    if 'current_page' in st.session_state and st.session_state.current_page == 'project_files' or st.session_state.page_index == 1:
        st.markdown('''
        <style>
            [data-testid="collapsedControl"]
                {
                display:block;
                }
        </style>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
                <style>
                    [data-testid="collapsedControl"]
                        {
                        display:none;
                        }
                </style>
                ''', unsafe_allow_html=True)

    selected2 = option_menu(None, menu,
                            icons=['house', 'cloud-upload', "list-task", 'gear'],
                            menu_icon="cast", default_index=st.session_state.page_index, orientation="horizontal",
                            styles={
                                "container": {"padding": "5px 20% 5px 20% !important", "background-color": "#86E5FF","width":"100vw",
                                              "border-radius": "0rem", "display": "block !important", "margin": "0px","z-index":"999991"}}
                            )
    st.markdown('''
             <style>
        [data-testid="stSidebarNav"]{display:none;}
        .css-91z34k,.css-k1ih3n{
        padding-top:27px;
        padding-left:7%;
        }
         </style>
            ''', unsafe_allow_html=True)
    menu_id = menu.index(selected2)
    if menu_id != st.session_state.page_index:

        st.session_state.page_index = menu_id
        if menu_id == 0:
            switch_page('home')
        elif menu_id == 3:
            switch_page('project')
        elif menu_id == 1:
            switch_page('demo')
        st._rerun()


def body_padding(len, px):
    st.markdown(f'''

    # <style>
    # .css-isz6o1:nth-child( {len} ) {{
    # margin:{px}px ;
    # }}
    # </style>
    ''', unsafe_allow_html=True)
