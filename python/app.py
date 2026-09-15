from pathlib import Path
import os

import altair as alt
import numpy as np
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph
from scipy import stats

from utils import load_json


st.set_page_config(page_title="DagFlow-AI", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

variables_path = Path(__file__).resolve().parent.parent / "synthdata" / "variables.json"

dag_path = Path(__file__).resolve().parent.parent / "synthdata" / "dag.json"

distributions_path = Path(__file__).resolve().parent.parent / "synthdata" / "distributions.json"

formulas_path = Path(__file__).resolve().parent.parent / "synthdata" / "formulas.json"
data_path = Path(__file__).resolve().parent.parent / "synthdata" / "generated_data.csv"
@st.cache_data
def load_csv(path: str, modified_at: float | None = None):
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    import pandas as pd

    return pd.read_csv(csv_path)


def prettify_text(value: str) -> str:
    if not isinstance(value, str):
        return str(value)

    pretty = value.replace("_", " ").replace("(", " ( ").replace(")", " ) ")
    pretty = " ".join(part for part in pretty.split() if part)
    return pretty.title()


def infer_column_order(df) -> list[str]:
    return list(df.columns)


def infer_numeric_columns(df, variables_data: dict | None = None) -> set[str]:
    numeric_columns = set(df.select_dtypes(include=[np.number]).columns)
    if not variables_data:
        return numeric_columns

    for column_name, metadata in variables_data.items():
        if metadata.get("measurement_level") == "Nominal" or metadata.get("measurement_level") == "Ordinal":
            numeric_columns.discard(column_name)
    return numeric_columns


def is_discrete_numeric_column(column_name: str, variables_data: dict | None = None) -> bool:
    if not variables_data or column_name not in variables_data:
        return False
    metadata = variables_data[column_name]
    return metadata.get("classification") == "Discrete"


def is_categorical_column(column_name: str, df, numeric_columns: set[str], variables_data: dict | None = None) -> bool:
    if variables_data and column_name in variables_data:
        measurement_level = variables_data[column_name].get("measurement_level")
        if measurement_level in {"Nominal", "Ordinal"}:
            return True
        if measurement_level == "Ratio" and variables_data[column_name].get("classification") in {"Discrete"}:
            return False

    return column_name not in numeric_columns


def get_domain_order(column_name: str, df, variables_data: dict | None = None, distributions_data: dict | None = None) -> list[str]:
    if distributions_data and column_name in distributions_data:
        categories = distributions_data[column_name].get("categories")
        if isinstance(categories, list) and categories:
            return [str(category) for category in categories]

    if variables_data and column_name in variables_data:
        metadata = variables_data[column_name]
        if metadata.get("measurement_level") in {"Nominal", "Ordinal"}:
            return list(dict.fromkeys(df[column_name].astype(str).tolist()))

    return list(dict.fromkeys(df[column_name].astype(str).tolist()))


def dataframe_value_counts(df, column_name: str, order: list[str]) -> list[dict]:
    counts = df[column_name].astype(str).value_counts()
    return [{"category": category, "count": int(counts.get(category, 0))} for category in order]


def build_univariate_chart(df, column_name: str, is_categorical: bool, is_discrete: bool, order: list[str] | None = None):
    if is_categorical:
        values = dataframe_value_counts(df, column_name, order or get_domain_order(column_name, df))
        return (
            alt.Chart(alt.Data(values=values))
            .mark_bar(color="#2E86DE")
            .encode(
                x=alt.X("category:N", title=prettify_text(column_name), sort=order or None),
                y=alt.Y("count:Q", title="Count"),
                tooltip=[alt.Tooltip("category:N", title="Category"), alt.Tooltip("count:Q", title="Count")],
            )
            .properties(height=360, title=prettify_text(column_name))
        )

    if is_discrete:
        values = dataframe_value_counts(df, column_name, order or get_domain_order(column_name, df))
        return (
            alt.Chart(alt.Data(values=values))
            .mark_bar(color="#2E86DE")
            .encode(
                x=alt.X("category:N", title=prettify_text(column_name), sort=order or None),
                y=alt.Y("count:Q", title="Count"),
                tooltip=[alt.Tooltip("category:N", title="Value"), alt.Tooltip("count:Q", title="Count")],
            )
            .properties(height=360, title=prettify_text(column_name))
        )

    values = df[column_name].dropna().astype(float)
    if values.empty:
        return None

    counts, bin_edges = np.histogram(values.to_numpy(), bins=30)
    histogram_data = [
        {
            "bin_start": float(bin_edges[idx]),
            "bin_end": float(bin_edges[idx + 1]),
            "bin_label": f"{bin_edges[idx]:.0f} - {bin_edges[idx + 1]:.0f}",
            "count": int(counts[idx]),
        }
        for idx in range(len(counts))
        if counts[idx] > 0
    ]

    return (
        alt.Chart(alt.Data(values=histogram_data))
        .mark_bar(color="#2E86DE")
        .encode(
            x=alt.X("bin_label:N", title=prettify_text(column_name), sort=None),
            y=alt.Y("count:Q", title="Count"),
            tooltip=[
                alt.Tooltip("bin_start:Q", title="Bin Start"),
                alt.Tooltip("bin_end:Q", title="Bin End"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=360, title=prettify_text(column_name))
    )


def build_bivariate_chart(df, x_name: str, y_name: str, x_is_categorical: bool, y_is_categorical: bool, variables_data: dict | None = None, distributions_data: dict | None = None):
    if not x_is_categorical and not y_is_categorical:
        return (
            alt.Chart(df)
            .mark_circle(color="#2E86DE", opacity=0.65, size=55)
            .encode(
                x=alt.X(f"{x_name}:Q", title=prettify_text(x_name)),
                y=alt.Y(f"{y_name}:Q", title=prettify_text(y_name)),
                tooltip=[alt.Tooltip(f"{x_name}:Q", title=prettify_text(x_name)), alt.Tooltip(f"{y_name}:Q", title=prettify_text(y_name))],
            )
            .properties(height=360, title=f"{prettify_text(x_name)} vs {prettify_text(y_name)}")
        )

    if x_is_categorical and y_is_categorical:
        x_order = get_domain_order(x_name, df, variables_data, distributions_data)
        y_order = get_domain_order(y_name, df, variables_data, distributions_data)
        chart_data = df.copy()
        chart_data[x_name] = chart_data[x_name].astype(str)
        chart_data[y_name] = chart_data[y_name].astype(str)
        return (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X(f"{x_name}:N", title=prettify_text(x_name), sort=x_order),
                xOffset=alt.XOffset(f"{y_name}:N", sort=y_order),
                y=alt.Y("count():Q", title="Count"),
                color=alt.Color(f"{y_name}:N", title=prettify_text(y_name), sort=y_order),
                tooltip=[alt.Tooltip(f"{x_name}:N", title=prettify_text(x_name)), alt.Tooltip(f"{y_name}:N", title=prettify_text(y_name)), alt.Tooltip("count():Q", title="Count")],
            )
            .properties(height=360, title=f"{prettify_text(x_name)} by {prettify_text(y_name)}")
        )

    categorical_name = x_name if x_is_categorical else y_name
    quantitative_name = y_name if x_is_categorical else x_name
    category_order = get_domain_order(categorical_name, df, variables_data, distributions_data)
    chart_data = df.copy()
    chart_data[categorical_name] = chart_data[categorical_name].astype(str)
    return (
        alt.Chart(chart_data)
        .mark_boxplot(color="#2E86DE")
        .encode(
            x=alt.X(f"{categorical_name}:N", title=prettify_text(categorical_name), sort=category_order),
            y=alt.Y(f"{quantitative_name}:Q", title=prettify_text(quantitative_name)),
            tooltip=[alt.Tooltip(f"{categorical_name}:N", title=prettify_text(categorical_name)), alt.Tooltip(f"{quantitative_name}:Q", title=prettify_text(quantitative_name))],
        )
        .properties(height=360, title=f"{prettify_text(quantitative_name)} by {prettify_text(categorical_name)}")
    )


def render_data_tab(df, variables_data: dict | None = None, distributions_data: dict | None = None) -> None:
    if data_path.exists():
        st.download_button(
            "Download generated data",
            data=data_path.read_bytes(),
            file_name=data_path.name,
            mime="text/csv",
            key="download_generated_data_button",
        )
    else:
        st.info("Generated data file is not available yet.")

    data_tabs = st.tabs(["Univariate", "Bivariate"])
    numeric_columns = infer_numeric_columns(df, variables_data)
    column_order = infer_column_order(df)

    with data_tabs[0]:
        st.subheader("Univariate")
        selected_column = st.selectbox("Select a variable", column_order, key="data_univariate_select")
        categorical = is_categorical_column(selected_column, df, numeric_columns, variables_data)
        discrete = is_discrete_numeric_column(selected_column, variables_data)
        order = get_domain_order(selected_column, df, variables_data, distributions_data) if categorical else None
        st.altair_chart(build_univariate_chart(df, selected_column, categorical, discrete, order), use_container_width=True)

    with data_tabs[1]:
        st.subheader("Bivariate")
        left_col, right_col = st.columns(2)
        with left_col:
            x_column = st.selectbox("Select x variable", column_order, key="data_bivariate_x")
        with right_col:
            y_column = st.selectbox("Select y variable", column_order, key="data_bivariate_y")

        x_categorical = is_categorical_column(x_column, df, numeric_columns, variables_data)
        y_categorical = is_categorical_column(y_column, df, numeric_columns, variables_data)
        st.altair_chart(
            build_bivariate_chart(df, x_column, y_column, x_categorical, y_categorical, variables_data, distributions_data),
            use_container_width=True,
        )

def render_details(selected_name: str, data: dict) -> None:
    item = data[selected_name]

    field_order = ["name", "description", "justification", "effect_type", "measurement_level", "classification", "skew"]

    for field_name in field_order:
        if field_name in item:
            left, right = st.columns([1, 3])
            left.write(prettify_text(field_name))
            right.write(prettify_text(item[field_name]))


def render_variable_tab(variables_data) -> None:
    variable_names = sorted(variables_data.keys())
    selected_label_to_name = {prettify_text(name): name for name in variable_names}
    selected_label = st.selectbox("Select a variable", sorted(selected_label_to_name.keys()), key="variable_select")
    variable_name = selected_label_to_name[selected_label]
    render_details(variable_name, variables_data)


def render_distribution_selector(distributions: dict) -> str:
    distribution_names = sorted(distributions.keys())
    selected_label_to_name = {prettify_text(name): name for name in distribution_names}
    selected_label = st.selectbox("Select a variable", sorted(selected_label_to_name.keys()), key="distribution_select")
    return selected_label_to_name[selected_label]


def render_category_probability_table(categories: list, probabilities: list) -> None:
    rows = []
    for category, probability in zip(categories, probabilities):
        rows.append({"Category": prettify_text(category), "Probability": format_numeric_value(probability)})
    st.table(rows)


def render_distribution_details(selected_name: str, distributions: dict) -> None:
    item = dict(distributions[selected_name])
    item["__name__"] = selected_name
    details_key = f"distribution_details_{selected_name}"
    left, right = st.columns([1, 1.4])

    with left:
        st.write(f"Distribution: {prettify_text(item.get('distribution', 'Unknown'))}")

        parameters = {key: value for key, value in item.items() if key != "distribution"}
        if not parameters:
            st.info("No parameters defined for this distribution.")
        else:
            for field_name, field_value in parameters.items():
                if field_name == "categories" and isinstance(field_value, list):
                    probabilities = parameters.get("probabilities", [])
                    render_category_probability_table(field_value, probabilities)
                    continue

                if field_name == "probabilities" and isinstance(field_value, list):
                    continue

                left_col, right_col = st.columns([1, 3])
                left_col.write(prettify_text(field_name))
                right_col.write(prettify_text(field_value))

    with right:
        with st.container(border=False):
            st.markdown(f"<div data-distribution-details='{details_key}'></div>", unsafe_allow_html=True)
            chart_placeholder = st.empty()
            chart = build_distribution_chart(item)
            if chart is None:
                chart_placeholder.info("No chart available for this distribution.")
            else:
                chart_placeholder.altair_chart(chart, use_container_width=True)


def render_field_value(field_name: str, field_value) -> None:
    left_col, right_col = st.columns([1, 1.4])
    label = "SNR" if field_name.lower() == "snr" else prettify_text(field_name)
    left_col.write(label)
    right_col.write(prettify_text(field_value))


def render_section_title(title: str) -> None:
    st.write(prettify_text(title))


def render_nested_value(value, indent: int = 0) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if isinstance(nested_value, (dict, list)):
                st.write(" " * indent + str(key))
                render_nested_value(nested_value, indent + 2)
            else:
                render_field_value(" " * indent + str(key), nested_value)
        return

    if isinstance(value, list):
        for idx, item in enumerate(value, start=1):
            if isinstance(item, (dict, list)):
                st.write(" " * indent + f"Item {idx}")
                render_nested_value(item, indent + 2)
            else:
                render_field_value(" " * indent + f"Item {idx}", item)
        return

    st.write(" " * indent + str(value))


def render_predictor_block(predictor_name: str, predictor: dict) -> None:
    if not predictor:
        st.info("No predictor details available.")
        return

    if "coefficient" in predictor:
        render_field_value("Coefficient", predictor.get("coefficient", "Unknown"))
        transformation = predictor.get("transformation")
        if transformation and transformation != "none":
            render_field_value("Transformation", transformation)
        return

    if "reference_category" in predictor and "other_categories" in predictor:
        render_field_value("Reference Category", predictor.get("reference_category", "Unknown"))
        other_categories = predictor.get("other_categories", {})
        if other_categories:
            rows = []
            for category_name, category_data in other_categories.items():
                rows.append(
                    {
                        "Other Categories": prettify_text(category_name),
                        "Coefficient": format_numeric_value(category_data.get("coefficient", "Unknown")),
                    }
                )
            st.table(rows)
        return

    for field_name, field_value in predictor.items():
        if isinstance(field_value, (dict, list)):
            render_section_title(field_name)
            render_nested_value(field_value)
        else:
            render_field_value(field_name, field_value)


def format_term_name(name: str, transformation: str | None = None) -> str:
    if transformation and transformation != "none":
        return f"{transformation}({name})"
    return name


def format_numeric_value(value) -> str:
    return f"{value:.3g}" if isinstance(value, (int, float)) else str(value)


def format_formula_terms(terms: list[str]) -> str:
    if not terms:
        return "0"

    formatted = [terms[0]]
    for term in terms[1:]:
        if term.startswith("-"):
            formatted.append(f"- {term[1:]}")
        else:
            formatted.append(f"+ {term}")
    return " ".join(formatted)


def format_formula_multiline(formula_text: str) -> str:
    parts = formula_text.split(" ~ ", 1)
    if len(parts) != 2:
        return formula_text

    left_side, right_side = parts
    if len(formula_text) <= 90:
        return formula_text

    terms = right_side.split(" + ")
    if len(terms) == 1:
        return formula_text

    lines = [f"{left_side} ~ {terms[0]}"]
    for term in terms[1:]:
        if term.startswith("- "):
            lines.append(f"  - {term[2:]}")
        else:
            lines.append(f"  + {term}")
    return "\n".join(lines)


def expand_predictors_for_display(predictors: dict) -> list[str]:
    terms = []
    for predictor_name, predictor in predictors.items():
        if "coefficient" in predictor:
            transformed_name = format_term_name(predictor_name, predictor.get("transformation"))
            terms.append(f"{transformed_name} * {format_numeric_value(predictor['coefficient'])}")
            continue

        if "reference_category" in predictor and "other_categories" in predictor:
            for category_name, category_data in predictor.get("other_categories", {}).items():
                terms.append(f"I({predictor_name} = {category_name}) * {format_numeric_value(category_data['coefficient'])}")
    return terms


def build_formula_string(response_name: str, formula: dict, distributions: dict, selected_category: str | None = None) -> str:
    distribution_name = distributions.get(response_name, {}).get("distribution", "")

    if "intercept" in formula and "predictors" in formula:
        terms = [format_numeric_value(formula.get("intercept", "0"))]
        terms.extend(expand_predictors_for_display(formula.get("predictors", {})))
        return f"{response_name} ~ " + format_formula_terms(terms)

    if "reference_category" in formula and "other_categories" in formula:
        category_name = selected_category or next(iter(formula.get("other_categories", {}).keys()), None)
        if category_name is None:
            return f"{response_name} ~ Unknown"

        category_block = formula["other_categories"][category_name]
        terms = [format_numeric_value(category_block.get("intercept", "0"))]
        terms.extend(expand_predictors_for_display(formula.get("predictors", {})))
        if "predictors" in category_block:
            terms.extend(expand_predictors_for_display(category_block.get("predictors", {})))
        return f"{response_name} ~ " + format_formula_terms(terms)

    return f"{response_name} ~ Unknown"


def render_formula_box(title: str, formula_text: str) -> None:
    st.subheader(title)
    st.code(format_formula_multiline(formula_text), language="text")


def render_section_header(title: str) -> None:
    st.markdown(f"<div style='margin-top: 0.75rem; margin-bottom: 0.35rem;'><h3 style='margin: 0;'>{prettify_text(title)}</h3></div>", unsafe_allow_html=True)
    st.divider()


def render_predictor_selector(predictors: dict, response_name: str, category_name: str | None = None) -> None:
    if not predictors:
        st.info("No predictors defined.")
        return

    predictor_names = sorted(predictors.keys())
    key_suffix = f"_{response_name}"
    if category_name is not None:
        key_suffix += f"_{category_name}"
    predictor_label_to_name = {prettify_text(name): name for name in predictor_names}
    selected_predictor_label = st.selectbox("Select a predictor", sorted(predictor_label_to_name.keys()), key=f"predictor_select{key_suffix}")
    selected_predictor = predictor_label_to_name[selected_predictor_label]
    render_predictor_block(selected_predictor, predictors[selected_predictor])


def render_quantitative_formula(response_name: str, formula: dict, distributions: dict) -> None:
    left, right = st.columns([1, 1.4])

    with left:
        render_section_header("General Info")
        render_field_value("Formula Type", "Quantitative")
        render_field_value("Intercept", formula.get("intercept", "Unknown"))
        render_field_value("Transformation", formula.get("transformation", "none"))
        render_field_value("SNR", formula.get("snr", "Unknown"))
        render_section_header("Predictors")
        render_predictor_selector(formula.get("predictors", {}), response_name)

    with right:
        render_formula_box("Formula", build_formula_string(response_name, formula, distributions))


def render_nominal_formula(response_name: str, formula: dict, distributions: dict) -> None:
    categories = formula.get("other_categories", {})
    category_names = sorted(categories.keys())

    left, right = st.columns([1, 1.4])

    with left:
        render_section_header("General Info")
        render_field_value("Formula Type", "Categorical Nominal")
        render_field_value("Reference Category", formula.get("reference_category", "Unknown"))

        if not category_names:
            st.info("No response categories defined.")
            return

        render_section_header("Category Details")
        category_label_to_name = {prettify_text(name): name for name in category_names}
        selected_category_label = st.selectbox(
            "Select a response category",
            sorted(category_label_to_name.keys()),
            key=f"nominal_category_select_{response_name}",
        )
        selected_category = category_label_to_name[selected_category_label]
        category_block = categories[selected_category]
        render_field_value("Intercept", category_block.get("intercept", "Unknown"))

        render_section_header("Predictors")
        render_predictor_selector(category_block.get("predictors", {}), response_name, selected_category)

    with right:
        render_formula_box("Formula", build_formula_string(response_name, formula, distributions, selected_category))


def render_ordinal_formula(response_name: str, formula: dict, distributions: dict) -> None:
    categories = formula.get("other_categories", {})
    category_names = sorted(categories.keys())

    left, right = st.columns([1, 1.4])

    with left:
        render_section_header("General Info")
        render_field_value("Formula Type", "Categorical Ordinal")
        render_field_value("Reference Category", formula.get("reference_category", "Unknown"))

        if not category_names:
            st.info("No threshold categories defined.")
            return

        render_section_header("Category Details")
        category_label_to_name = {prettify_text(name): name for name in category_names}
        selected_category_label = st.selectbox(
            "Select a response category",
            sorted(category_label_to_name.keys()),
            key=f"ordinal_category_select_{response_name}",
        )
        selected_category = category_label_to_name[selected_category_label]
        render_field_value("Intercept", categories[selected_category].get("intercept", "Unknown"))

        render_section_header("Predictors")
        render_predictor_selector(formula.get("predictors", {}), response_name)

    with right:
        render_formula_box("Formula", build_formula_string(response_name, formula, distributions, selected_category))


def render_formula_block(response_name: str, formula: dict, distributions: dict) -> None:
    if not formula:
        st.info("No formula defined for this variable.")
        return

    if "intercept" in formula and "predictors" in formula:
        render_quantitative_formula(response_name, formula, distributions)
        return

    if "reference_category" in formula and "other_categories" in formula and "predictors" in formula and "intercept" not in formula:
        render_ordinal_formula(response_name, formula, distributions)
        return

    if "reference_category" in formula and "other_categories" in formula:
        render_nominal_formula(response_name, formula, distributions)
        return

    st.info("Unsupported formula format.")


def render_formulas_tab(formulas_data, distributions_data) -> None:
    formulas_by_name = formulas_data
    response_names = sorted(formulas_by_name.keys())
    response_label_to_name = {prettify_text(name): name for name in response_names}
    selected_response_label = st.selectbox("Select a variable", sorted(response_label_to_name.keys()), key="formula_select")
    response_name = response_label_to_name[selected_response_label]
    render_formula_block(response_name, formulas_by_name[response_name], distributions_data)


def build_distribution_chart(item: dict):
    distribution_name = item.get("distribution", "")
    parameters = {key: value for key, value in item.items() if key != "distribution"}
    chart_title = item.get("__name__")

    if distribution_name in {"Normal", "Exponential", "Gamma", "Log Normal", "Beta", "Uniform"}:
        return build_truncated_density_chart(distribution_name, parameters, chart_title)

    if distribution_name in {"Discrete Uniform", "Bernoulli", "Binomial", "Poisson", "Geometric", "Negative Binomial"}:
        return build_truncated_pmf_chart(distribution_name, parameters, chart_title)

    if distribution_name in {"Categorical Ordinal", "Categorical Nominal"}:
        return build_categorical_chart(parameters, chart_title)

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


def build_truncated_density_chart(distribution_name: str, parameters: dict, chart_title: str | None = None):
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
        .properties(height=320, title=chart_title or distribution_name)
    )


def build_truncated_pmf_chart(distribution_name: str, parameters: dict, chart_title: str | None = None):
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
        .properties(height=320, title=chart_title or distribution_name)
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
        if "min" not in parameters or "max" not in parameters:
            return None
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
        if "min" not in parameters or "max" not in parameters:
            return np.array([])
        return np.arange(int(parameters["min"]), int(parameters["max"]) + 1)
    if distribution_name == "Bernoulli":
        return np.array([0, 1])
    if distribution_name == "Binomial":
        return np.arange(0, int(parameters["n_trials"]) + 1)
    if distribution_name == "Poisson":
        upper_bound = max_value if max_value is not None else parameters.get("max")
        upper = int(max(upper_bound if upper_bound is not None else 0, parameters["rate"] + 6 * np.sqrt(parameters["rate"])))
        return np.arange(0, upper + 1)
    if distribution_name == "Geometric":
        upper_bound = max_value if max_value is not None else parameters.get("max")
        upper = int(max(upper_bound if upper_bound is not None else 25, 25))
        return np.arange(max(1, int(parameters.get("min", 1))), upper + 1)
    if distribution_name == "Negative Binomial":
        shape = parameters["shape"]
        mean = parameters["mean"]
        upper_bound = max_value if max_value is not None else parameters.get("max")
        upper = int(max(upper_bound if upper_bound is not None else 0, mean + 6 * np.sqrt(mean + mean * mean / shape)))
        return np.arange(max(0, int(parameters.get("min", 0))), upper + 1)
    return np.array([])


def build_categorical_chart(parameters: dict, chart_title: str | None = None):
    categories = parameters.get("categories", [])
    probabilities = parameters.get("probabilities", [])
    if not categories or not probabilities:
        return None

    values = [
        {"category": prettify_text(category), "probability": float(probability)}
        for category, probability in zip(categories, probabilities)
    ]
    return (
        alt.Chart(alt.Data(values=values))
        .mark_bar(color="#2E86DE")
        .encode(
            x=alt.X("category:N", title="Category", sort=None),
            y=alt.Y("probability:Q", title="Probability"),
            tooltip=[alt.Tooltip("category:N", title="Category"), alt.Tooltip("probability:Q", title="Probability")],
        )
        .properties(height=320, title=chart_title or "Categories")
    )


def render_dag(dag_data: dict) -> None:
    nodes = []
    for node_id, node_data in dag_data["nodes"].items():
        node_type = node_data.get("type", "stochastic")
        color = "#2E86DE" if node_type == "stochastic" else "#7F8C8D"
        nodes.append(
            Node(
                id=node_id,
                label=prettify_text(node_id),
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
        width=1400,
        height=850,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        node={"size": 28},
        randomSeed=7,
    )

    agraph(nodes=nodes, edges=edges, config=config)


def render_selected_section() -> None:
    if "selected_section" not in st.session_state:
        st.session_state.selected_section = "Chat"

    section_names = ["Chat", "Variables", "DAG", "Distributions", "Formulas", "Data"]
    left_nav, main_panel = st.columns([1, 5], gap="large")

    with left_nav:
        st.subheader("Sections")
        st.radio(
            "Navigate sections",
            section_names,
            key="selected_section",
            label_visibility="collapsed",
        )

    with main_panel:
        if st.session_state.selected_section == "Chat":
            st.components.v1.iframe(chat_url, height=600, scrolling=True)
        elif st.session_state.selected_section == "Variables":
            st.header("Variables")
            variables_data = load_json(variables_path)
            if variables_data is None:
                st.info("Variables have not been created yet.")
            else:
                render_variable_tab(variables_data)
        elif st.session_state.selected_section == "DAG":
            st.header("DAG")
            dag_data = load_json(dag_path)
            if dag_data is None:
                st.info("DAG has not been created yet.")
            else:
                render_dag(dag_data)
        elif st.session_state.selected_section == "Distributions":
            st.header("Distributions")
            distributions_data = load_json(distributions_path)
            if distributions_data is None:
                st.info("Distributions have not been created yet.")
            else:
                distributions = distributions_data
                distribution_name = render_distribution_selector(distributions)
                render_distribution_details(distribution_name, distributions)
        elif st.session_state.selected_section == "Formulas":
            st.header("Formulas")
            formulas_data = load_json(formulas_path)
            distributions_data = load_json(distributions_path) or {}
            if formulas_data is None:
                st.info("Formulas have not been created yet.")
            else:
                render_formulas_tab(formulas_data, distributions_data)
        elif st.session_state.selected_section == "Data":
            st.header("Data")
            data_mtime = data_path.stat().st_mtime if data_path.exists() else None
            data_df = load_csv(str(data_path), data_mtime)
            variables_data = load_json(variables_path) or {}
            distributions_data = load_json(distributions_path) or {}
            if data_df is None:
                st.info("Generated data has not been created yet.")
            else:
                render_data_tab(data_df, variables_data, distributions_data)


st.title("DagFlow")
opencode_port = int(os.environ.get("OPENCODE_PORT", "4096"))
codespace_name = os.environ.get("CODESPACE_NAME")
if codespace_name:
    chat_url = f"https://{codespace_name}-{opencode_port}.app.github.dev"
else:
    chat_url = f"http://127.0.0.1:{opencode_port}"
render_selected_section()
