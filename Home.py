
import streamlit as st

st.set_page_config(page_title="Tools Home", page_icon="🏠", layout="wide")
st.title("PTL Cable Tools Home")

st.write("Quick links to all pages:")
st.page_link("Home.py", label="Home", icon="🏠")
st.page_link("pages/Paradise_Summary.py", label="Paradise Summary")
st.page_link("pages/Paradise_Tools.py", label="Paradise Tools")
st.page_link("pages/Paradise_Tools_few_clicks.py", label="Paradise Tools (operator version)")
st.page_link("pages/Tesla_Summary.py", label="Tesla Summary")
st.page_link("pages/Tesla_Tools.py", label="Tesla Tools")
st.page_link("pages/Tesla_Tools_few_clicks.py", label="Tesla Tools (operator version)")

