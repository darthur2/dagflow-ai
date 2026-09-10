import json
from pathlib import Path

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph


st.set_page_config(page_title="DagFlow", layout="wide")

variables_path = Path(__file__).resolve().parent.parent / "synthdata" / "variables.json"

dag_path = Path(__file__).resolve().parent.parent / "synthdata" / "dag.json"

distributions_path = Path(__file__).resolve().parent.parent / "synthdata" / "distributions.json"


def load_json(path: Path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

st.title("DagFlow")

tabs = st.tabs(["Variables", "DAG", "Distributions", "Formulas", "Data"])


def render_details(selected_name: str, data: dict) -> None:
    item = data[selected_name]
    st.subheader(selected_name)

    field_order = ["description", "justification", "effect_type", "measurement_level"]
    if "classification" in item:
        field_order.extend(["classification", "skew"])

    for field_name in field_order:
        if field_name in item:
            left, right = st.columns([1, 3])
            left.write(field_name)
            right.write(item[field_name])


def render_distribution_details(selected_name: str, distributions: dict) -> None:
    item = distributions[selected_name]
    st.subheader(selected_name)
    st.write(f"Distribution: {item.get('distribution', 'Unknown')}")

    parameters = item.get("parameters", {})
    if not parameters:
        st.info("No parameters defined for this distribution.")
        return

    st.write("Parameters")
    for field_name, field_value in parameters.items():
        left, right = st.columns([1, 3])
        left.write(field_name)
        right.write(field_value)


def render_dag(dag_data: dict) -> None:
    nodes = []
    for node_id, node_data in dag_data["nodes"].items():
        node_type = node_data.get("type", "stochastic")
        color = "#2E86DE" if node_type == "stochastic" else "#7F8C8D"
        nodes.append(
            Node(
                id=node_id,
                label=node_id,
                size=18,
                color=color,
                font={"color": "#FFFFFF"},
            )
        )

    edges = []
    for edge_data in dag_data["edges"].values():
        edges.append(
            Edge(
                source=edge_data["parent"],
                target=edge_data["child"],
            )
        )

    config = Config(
        width=1200,
        height=800,
        directed=True,
        physics=True,
        hierarchical=True,
        nodeHighlightBehavior=True,
    )

    agraph(nodes=nodes, edges=edges, config=config)

with tabs[0]:
    st.header("Variables")
    variables_data = load_json(variables_path)
    if variables_data is None:
        st.info("Variables have not been created yet.")
    else:
        variable_names = {item["name"]: item for item in variables_data}
        variable_name = st.selectbox("Select a variable", sorted(variable_names.keys()), key="variable_select")
        render_details(variable_name, variable_names)

with tabs[1]:
    st.header("DAG")
    dag_data = load_json(dag_path)
    if dag_data is None:
        st.info("DAG has not been created yet.")
    else:
        render_dag(dag_data["dag"])

with tabs[2]:
    st.header("Distributions")
    distributions_data = load_json(distributions_path)
    if distributions_data is None:
        st.info("Distributions have not been created yet.")
    else:
        distributions = distributions_data
        distribution_name = st.selectbox("Select a variable", sorted(distributions.keys()), key="distribution_select")
        render_distribution_details(distribution_name, distributions)

with tabs[3]:
    st.header("Formulas")
    formulas_path = Path(__file__).resolve().parent.parent / "synthdata" / "formulas.json"
    formulas_data = load_json(formulas_path)
    if formulas_data is None:
        st.info("Formulas have not been created yet.")
    else:
        st.write("Formulas loaded.")

with tabs[4]:
    st.header("Data")
    st.write("Data tab coming soon.")
