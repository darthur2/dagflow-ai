import json
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="DagFlow", layout="wide")

variables_path = Path(__file__).resolve().parent.parent / "synthdata" / "variables.json"
with variables_path.open("r", encoding="utf-8") as f:
    variables = json.load(f)["variables"]

st.title("DagFlow")
st.write("Streamlit app skeleton")

tabs = st.tabs(["Variables", "DAG", "Distributions", "Formulas", "Data"])


def render_details(selected_name: str, data: dict) -> None:
    item = data[selected_name]
    st.subheader(selected_name)

    for field_name, value in item.items():
        left, right = st.columns([1, 3])
        left.write(field_name)
        right.write(value)

with tabs[0]:
    st.header("Variables")
    variable_name = st.selectbox("Select a variable", sorted(variables.keys()), key="variable_select")
    render_details(variable_name, variables)

with tabs[1]:
    st.header("DAG")
    st.write("DAG tab coming soon.")

with tabs[2]:
    st.header("Distributions")
    distributions_path = Path(__file__).resolve().parent.parent / "synthdata" / "distributions.json"
    with distributions_path.open("r", encoding="utf-8") as f:
        distributions = json.load(f)["distributions"]

    distribution_name = st.selectbox("Select a variable", sorted(distributions.keys()), key="distribution_select")
    render_details(distribution_name, distributions)

with tabs[3]:
    st.header("Formulas")
    st.write("Formulas tab coming soon.")

with tabs[4]:
    st.header("Data")
    st.write("Data tab coming soon.")
