import pandas as pd
import streamlit as st

from modules import utils
from modules.dataframe import display, info, stats, correlation, duplicate, group
def ds_analysis(ds):
	try:
		data=ds.file_data
		data_opt=0
	except KeyError:
		st.header("No Dataset Found")
		st.stop()

	except Exception as e:
		st.write(e)
		st.stop()

	menus = ["Display", "Information", "Statistics", "Correlation", "Duplicate", "Group"]
	tabs = st.tabs(menus)

	with tabs[0]:
		display.display(data)

	with tabs[1]:
		info.info(data)

	with tabs[2]:
		stats.stats(data)

	with tabs[3]:
		correlation.correlation(data)

	with tabs[4]:
		duplicate.duplicate(data, data_opt)

	with tabs[5]:
		group.group(data)
