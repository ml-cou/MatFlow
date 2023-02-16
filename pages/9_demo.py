import streamlit as st
from pathlib import Path
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
from subpage.navbar import navbar,body_padding
navbar()
base_path = Path().absolute()
from streamlit_option_menu import option_menu
from epsilon import Deep4che,Dynamocs,Photoche,Pubche,Experimental,Training,Unknown,Overall_Randomforest_Classifaications,Ovall_RandomForest_Regrassion,OverallData,Prediction

ontions_ = ['Data Extraction', 'Build Dataset', 'Machine Learning']

de = ['Deep4chem', 'Dyomics', 'Photochem', 'Pubchem']
bd = ['Experimental', 'Training', 'Unknown']
ml=['Random Forest Classifier', 'Random Forest Regression', 'Overall Data','Prediction']
tmp = ml
show_nav=True
st.markdown('''
        <style>
        [data-testid="stSidebarNav"]{display:none;}
        # .css-91z34k,.css-k1ih3n{
        # padding-top:27px;
        # padding-left:7%;
        # }
        </style>
        ''', unsafe_allow_html=True)
with st.sidebar:
    selected = option_menu('Dye Design', ontions_, menu_icon="", icons='',
                           styles={
                               "container": {"padding-top": "10px !important", "background-color": "transparent"}}
                           )
st.markdown('''
            <style>
            .css-110bie7
            {
            min-width: 200px !important;
            max-width: 400px !important;
            }
            [data-testid="stExpander"]
            {
            background-color:white;
            }
            </style>
            ''', unsafe_allow_html=True)
c0,col1,c1=st.columns([.5,6,.5])
with col1:
    if selected == ontions_[0]:
        tmp = de
    elif selected == ontions_[1]:
        tmp = bd
    if show_nav:
        selected2 = option_menu(None, tmp,
                                icons=['caret-down-fill']*len(tmp),

                                menu_icon="cast", default_index=0, orientation="horizontal",
                                styles={
                                    "container": {"background-color": "transparent"}}
                                )
        if selected2==de[0]:
            Deep4che.deep4che()
        elif selected2==de[1]:
            Dynamocs.dynamocs()
        elif selected2==de[2]:
            Photoche.photoche()
        elif selected2==de[3]:
            Pubche.pubche()
        elif selected2==bd[0]:
            Experimental.experimental()
        elif selected2==bd[1]:
            Training.training()
        elif selected2==bd[2]:
                Unknown.unknown()
        elif selected2==ml[0]:
            Overall_Randomforest_Classifaications.ovall_randomforest_classifaications()
        elif selected2==ml[1]:
            Ovall_RandomForest_Regrassion.ovall_randomforest_regrassion()
        elif selected2==ml[2]:
            OverallData.overalldata()
        else:
            Prediction.prediction()

