import json
from pathlib import Path

import altair as alt
import numpy as np
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph
from scipy import stats


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
    left, right = st.columns([1, 1.4])

    with left:
        st.subheader(selected_name)
        st.write(f"Distribution: {item.get('distribution', 'Unknown')}")

        parameters = item.get("parameters", {})
        if not parameters:
            st.info("No parameters defined for this distribution.")
        else:
            st.write("Parameters")
            for field_name, field_value in parameters.items():
                left_col, right_col = st.columns([1, 3])
                left_col.write(field_name)
                right_col.write(field_value)

    with right:
        chart = build_distribution_chart(item)
        if chart is None:
            st.info("No chart available for this distribution.")
        else:
            st.altair_chart(chart, use_container_width=True)


def build_distribution_chart(item: dict):
    distribution_name = item.get("distribution", "")
    parameters = item.get("parameters", {})

    if distribution_name in {"Normal", "Exponential", "Gamma", "Log Normal", "Beta", "Uniform"}:
        return build_truncated_density_chart(distribution_name, parameters)

    if distribution_name in {"Discrete Uniform", "Bernoulli", "Binomial", "Poisson", "Geometric", "Negative Binomial"}:
        return build_truncated_pmf_chart(distribution_name, parameters)

    if distribution_name in {"Categorical Ordinal", "Categorical Nominal"}:
        return build_categorical_chart(parameters)

    return None


def get_bounds(parameters: dict):
    return parameters.get("min"), parameters.get("max")


def apply_min_max_mask(values: np.ndarray, min_value, max_value):
    mask = np.ones_like(values, dtype=bool)
    if min_value is not None:
        mask &= values >= min_value
    if max_value is not None:
        mask &= values <= max_value
    return mask


def build_truncated_density_chart(distribution_name: str, parameters: dict):
    min_value, max_value = get_bounds(parameters)
    dist = density_distribution(distribution_name, parameters)
    if dist is None:
        return None

    min_value = min_value if min_value is not None else support_floor(distribution_name, dist)
    max_value = max_value if max_value is not None else support_ceiling(distribution_name, dist)

    x = np.linspace(min_value, max_value, 400)
    y = dist.pdf(x)
    truncation_mass = dist.cdf(max_value) - dist.cdf(min_value)
    if truncation_mass <= 0:
        return None
    y = y / truncation_mass

    density_data = {"x": x, "density": y}
    return (
        alt.Chart(alt.Data(values=[{"x": float(xi), "density": float(yi)} for xi, yi in zip(density_data["x"], density_data["density"])]))
        .mark_line(color="#2E86DE")
        .encode(
            x=alt.X("x:Q", title=distribution_name),
            y=alt.Y("density:Q", title="Density"),
            tooltip=[alt.Tooltip("x:Q", title="x"), alt.Tooltip("density:Q", title="density")],
        )
        .properties(height=320)
    )


def build_truncated_pmf_chart(distribution_name: str, parameters: dict):
    min_value, max_value = get_bounds(parameters)

    dist = pmf_distribution(distribution_name, parameters)
    if dist is None:
        return None

    support = pmf_support(distribution_name, parameters, min_value, max_value)
    probs = dist.pmf(support)

    mask = apply_min_max_mask(support, min_value, max_value)
    support = support[mask]
    probs = probs[mask]
    total_mass = probs.sum()
    if total_mass <= 0:
        return None
    probs = probs / total_mass

    pmf_values = [{"x": int(xi), "probability": float(pi)} for xi, pi in zip(support, probs)]
    return (
        alt.Chart(alt.Data(values=pmf_values))
        .mark_bar(color="#2E86DE")
        .encode(
            x=alt.X("x:O", title=distribution_name),
            y=alt.Y("probability:Q", title="Probability"),
            tooltip=[alt.Tooltip("x:O", title="x"), alt.Tooltip("probability:Q", title="probability")],
        )
        .properties(height=320)
    )


def density_distribution(distribution_name: str, parameters: dict):
    if distribution_name == "Normal":
        return stats.norm(loc=parameters["mean"], scale=parameters["standard_deviation"])
    if distribution_name == "Exponential":
        return stats.expon(scale=1.0 / parameters["rate"])
    if distribution_name == "Gamma":
        return stats.gamma(a=parameters["shape"], scale=1.0 / parameters["rate"])
    if distribution_name == "Log Normal":
        return stats.lognorm(s=parameters["log_standard_deviation"], scale=np.exp(parameters["log_mean"]))
    if distribution_name == "Beta":
        return stats.beta(a=parameters["shape_1"], b=parameters["shape_2"])
    if distribution_name == "Uniform":
        return stats.uniform(loc=parameters["min"], scale=parameters["max"] - parameters["min"])
    return None


def support_floor(distribution_name: str, dist):
    if distribution_name in {"Exponential", "Gamma", "Uniform"}:
        return 0 if distribution_name != "Uniform" else dist.support()[0]
    return dist.ppf(0.001)


def support_ceiling(distribution_name: str, dist):
    if distribution_name == "Uniform":
        return dist.support()[1]
    return dist.ppf(0.999)


def pmf_distribution(distribution_name: str, parameters: dict):
    if distribution_name == "Discrete Uniform":
        return stats.randint(low=parameters["min"], high=parameters["max"] + 1)
    if distribution_name == "Bernoulli":
        return stats.bernoulli(p=parameters["success_prob"])
    if distribution_name == "Binomial":
        return stats.binom(n=parameters["n_trials"], p=parameters["success_prob"])
    if distribution_name == "Poisson":
        return stats.poisson(mu=parameters["rate"])
    if distribution_name == "Geometric":
        return stats.geom(p=parameters["success_prob"])
    if distribution_name == "Negative Binomial":
        shape = parameters["shape"]
        mean = parameters["mean"]
        p = shape / (shape + mean)
        return stats.nbinom(n=shape, p=p)
    return None


def pmf_support(distribution_name: str, parameters: dict, min_value, max_value):
    if distribution_name == "Discrete Uniform":
        return np.arange(int(parameters["min"]), int(parameters["max"]) + 1)
    if distribution_name == "Bernoulli":
        return np.array([0, 1])
    if distribution_name == "Binomial":
        return np.arange(0, int(parameters["n_trials"]) + 1)
    if distribution_name == "Poisson":
        upper = int(max(parameters["max"] if max_value is None else max_value, parameters["rate"] + 6 * np.sqrt(parameters["rate"])))
        return np.arange(0, upper + 1)
    if distribution_name == "Geometric":
        upper = int(max(parameters["max"] if max_value is None else max_value, 25))
        return np.arange(max(1, int(parameters.get("min", 1))), upper + 1)
    if distribution_name == "Negative Binomial":
        shape = parameters["shape"]
        mean = parameters["mean"]
        upper = int(max(parameters["max"] if max_value is None else max_value, mean + 6 * np.sqrt(mean + mean * mean / shape)))
        return np.arange(max(0, int(parameters.get("min", 0))), upper + 1)
    return np.array([])


def build_categorical_chart(parameters: dict):
    categories = parameters.get("categories", [])
    probabilities = parameters.get("probabilities", [])
    if not categories or not probabilities:
        return None

    values = [
        {"category": category, "probability": float(probability)}
        for category, probability in zip(categories, probabilities)
    ]
    return (
        alt.Chart(alt.Data(values=values))
        .mark_bar(color="#2E86DE")
        .encode(
            x=alt.X("category:N", title="Category", sort=None),
            y=alt.Y("probability:Q", title="Probability"),
            tooltip=[alt.Tooltip("category:N", title="category"), alt.Tooltip("probability:Q", title="probability")],
        )
        .properties(height=320)
    )


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
