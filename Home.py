
import streamlit as st

st.set_page_config(page_title="Tools Home", layout="wide")
st.title("PTL Cable Tools Home")

st.write("Quick links to all pages:")
st.page_link("Home.py", label="Home")
st.page_link("pages/paradise_summary.py", label="Paradise Summary")
st.page_link("pages/paradise_tools.py", label="Paradise Tools")
st.page_link("pages/paradise_tools_few_clicks.py", label="Paradise Tools (operator version)")
st.page_link("pages/tesla_summary.py", label="Tesla Summary")
st.page_link("pages/tesla_tools.py", label="Tesla Tools")
st.page_link("pages/tesla_tools_few_clicks.py", label="Tesla Tools (operator version)")

