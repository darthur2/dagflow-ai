import streamlit as st


st.set_page_config(page_title="DagFlow", layout="wide")

st.title("DagFlow")
st.write("Streamlit app skeleton")

tabs = st.tabs(["Variables", "DAG", "Distributions", "Formulas", "Data"])

with tabs[0]:
    st.header("Variables")
    st.write("Variables tab coming soon.")

with tabs[1]:
    st.header("DAG")
    st.write("DAG tab coming soon.")

with tabs[2]:
    st.header("Distributions")
    st.write("Distributions tab coming soon.")

with tabs[3]:
    st.header("Formulas")
    st.write("Formulas tab coming soon.")

with tabs[4]:
    st.header("Data")
    st.write("Data tab coming soon.")
