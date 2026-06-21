plot_dag_visnetwork <- function(nodes, edges) {
  if (!is.list(nodes) || length(nodes) == 0) {
    stop("`nodes` must be a non-empty list")
  }
  if (!is.list(edges)) {
    stop("`edges` must be a list")
  }

  get_type <- function(n) {
    if (is.null(n$type)) "endogenous" else n$type
  }

  nodes_df <- data.frame(
    id = vapply(nodes, `[[`, character(1), "id"),
    label = vapply(nodes, `[[`, character(1), "id"),
    group = vapply(nodes, get_type, character(1)),
    title = vapply(nodes, function(n) {
      if (!is.null(n$description)) n$description else n$id
    }, character(1)),
    stringsAsFactors = FALSE
  )

  if (nrow(nodes_df) == 0) {
    stop("No nodes to plot")
  }

  edges_df <- data.frame(
    from = vapply(edges, `[[`, character(1), "from"),
    to = vapply(edges, `[[`, character(1), "to"),
    arrows = "to",
    stringsAsFactors = FALSE
  )

  visNetwork::visNetwork(nodes_df, edges_df) |>
    visNetwork::visGroups(
      groupname = "endogenous",
      color = list(
        background = "#85C1E9",
        border = "#2E86C1",
        highlight = list(background = "#3498DB", border = "#2980B9")
      )
    ) |>
    visNetwork::visGroups(
      groupname = "exogenous",
      color = list(
        background = "#F1948A",
        border = "#C0392B",
        highlight = list(background = "#E74C3C", border = "#922B21")
      )
    ) |>
    visNetwork::visOptions(
      highlightNearest = list(enabled = TRUE, degree = 1, hover = TRUE),
      nodesIdSelection = TRUE
    ) |>
    visNetwork::visPhysics(
      solver = "forceAtlas2Based",
      stabilization = list(iterations = 200)
    ) |>
    visNetwork::visLayout(randomSeed = 42) |>
    visNetwork::visLegend(
      useGroups = TRUE,
      position = "right",
      stepX = 100
    ) |>
    visNetwork::visInteraction(
      dragNodes = TRUE,
      dragView = TRUE,
      zoomView = TRUE,
      navigationButtons = TRUE
    )
}
