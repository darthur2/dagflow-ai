if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Package 'jsonlite' is required")
}

detect_cycle_nodes <- function(nodes, edges) {
  node_ids <- vapply(nodes, `[[`, character(1), "id")

  adj <- list()
  in_degree <- integer(length(node_ids))
  names(in_degree) <- node_ids

  for (id in node_ids) {
    adj[[id]] <- character(0)
    in_degree[id] <- 0L
  }

  for (e in edges) {
    from <- e$from
    to <- e$to
    adj[[from]] <- c(adj[[from]], to)
    in_degree[to] <- in_degree[to] + 1L
  }

  queue <- node_ids[in_degree == 0L]
  processed <- character(0)

  while (length(queue) > 0) {
    current <- queue[1]
    queue <- queue[-1]
    processed <- c(processed, current)

    for (neighbor in adj[[current]]) {
      in_degree[neighbor] <- in_degree[neighbor] - 1L
      if (in_degree[neighbor] == 0L) {
        queue <- c(queue, neighbor)
      }
    }
  }

  cycle_nodes <- setdiff(node_ids, processed)
  cycle_nodes
}

validate_dag <- function(variables_path = "synthdata/variables.json",
                          dag_path = "synthdata/dag.json",
                          output_path = "synthdata/dag_validation_result.json") {
  variables <- jsonlite::fromJSON(variables_path, simplifyVector = FALSE)
  dag <- jsonlite::fromJSON(dag_path, simplifyVector = FALSE)

  errors <- list()

  if (!is.list(variables) || length(variables) == 0) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "root",
      issue = "variables.json must contain a non-empty array"
    )
    result <- list(valid = FALSE, n_variables = 0, errors = errors)
    jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    return(invisible(result))
  }

  if (!is.list(dag)) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "root",
      issue = "dag.json must contain a valid JSON object"
    )
    result <- list(valid = FALSE, n_variables = length(variables), errors = errors)
    jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    return(invisible(result))
  }

  if (is.null(dag$nodes)) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "nodes",
      issue = "missing required field 'nodes' in dag.json"
    )
  }

  if (is.null(dag$edges)) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "edges",
      issue = "missing required field 'edges' in dag.json"
    )
  }

  if (is.null(dag$nodes) || is.null(dag$edges)) {
    result <- list(valid = FALSE, n_variables = length(variables), errors = errors)
    jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    return(invisible(result))
  }

  dag_nodes <- dag$nodes
  dag_edges <- dag$edges

  if (!is.list(dag_nodes)) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "nodes",
      issue = "'nodes' must be an array"
    )
  }

  if (!is.list(dag_edges)) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "edges",
      issue = "'edges' must be an array"
    )
  }

  if (!is.list(dag_nodes) || !is.list(dag_edges)) {
    result <- list(valid = FALSE, n_variables = length(variables), errors = errors)
    jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    return(invisible(result))
  }

  var_names <- vapply(variables, `[[`, character(1), "name")
  
  dag_node_ids <- character(0)
  for (n in dag_nodes) {
    if (is.null(n$id)) {
      errors[[length(errors) + 1]] <- list(
        variable = NA, field = "id",
        issue = "a node is missing the required 'id' field"
      )
      next
    }
    dag_node_ids <- c(dag_node_ids, n$id)

    if (is.null(n$type)) {
      errors[[length(errors) + 1]] <- list(
        variable = n$id, field = "type",
        issue = sprintf("node '%s' is missing the required 'type' field", n$id)
      )
    } else if (!n$type %in% c("endogenous", "exogenous")) {
      errors[[length(errors) + 1]] <- list(
        variable = n$id, field = "type",
        issue = sprintf("node '%s' has invalid type '%s'; must be 'endogenous' or 'exogenous'",
                        n$id, n$type)
      )
    }

    if (!is.null(n$type) && n$type == "exogenous" && is.null(n$description)) {
      errors[[length(errors) + 1]] <- list(
        variable = n$id, field = "description",
        issue = sprintf("exogenous node '%s' is missing required 'description' field", n$id)
      )
    }

    if (!n$id %in% var_names) {
      errors[[length(errors) + 1]] <- list(
        variable = n$id, field = "id",
        issue = sprintf("node '%s' has no matching variable in variables.json", n$id)
      )
    }
  }

  for (v_name in var_names) {
    if (!v_name %in% dag_node_ids) {
      errors[[length(errors) + 1]] <- list(
        variable = v_name, field = "id",
        issue = sprintf("variable '%s' from variables.json has no node in the DAG", v_name)
      )
    }
  }

  for (i in seq_along(dag_edges)) {
    e <- dag_edges[[i]]
    edge_label <- sprintf("edge %d", i)

    if (is.null(e$from) || is.null(e$to)) {
      errors[[length(errors) + 1]] <- list(
        variable = NA, field = "edges",
        issue = sprintf("%s is missing required 'from' or 'to' field", edge_label)
      )
      next
    }

    extra_edge_fields <- setdiff(names(e), c("from", "to"))
    if (length(extra_edge_fields) > 0) {
      errors[[length(errors) + 1]] <- list(
        variable = NA, field = "edges",
        issue = sprintf("%s (from '%s' to '%s') has unexpected field '%s'; edges must have only 'from' and 'to'",
                        edge_label, e$from, e$to, extra_edge_fields[1])
      )
    }

    if (!e$from %in% var_names) {
      errors[[length(errors) + 1]] <- list(
        variable = e$from, field = "edges",
        issue = sprintf("edge 'from' value '%s' not found in variable list", e$from)
      )
    }

    if (!e$to %in% var_names) {
      errors[[length(errors) + 1]] <- list(
        variable = e$to, field = "edges",
        issue = sprintf("edge 'to' value '%s' not found in variable list", e$to)
      )
    }
  }

  incoming_edges <- list()
  for (id in dag_node_ids) {
    incoming_edges[[id]] <- character(0)
  }
  for (e in dag_edges) {
    if (!is.null(e$to) && e$to %in% dag_node_ids) {
      incoming_edges[[e$to]] <- c(incoming_edges[[e$to]], e$from)
    }
  }

  for (n in dag_nodes) {
    nid <- n$id
    ntype <- n$type
    n_parents <- incoming_edges[[nid]]

    if (is.null(ntype)) next

    if (ntype == "endogenous" && length(n_parents) == 0) {
      errors[[length(errors) + 1]] <- list(
        variable = nid, field = "type",
        issue = sprintf("node '%s' has 0 incoming edges but is labeled 'endogenous'; must be 'exogenous'",
                        nid)
      )
    }

    if (ntype == "exogenous" && length(n_parents) > 0) {
      errors[[length(errors) + 1]] <- list(
        variable = nid, field = "type",
        issue = sprintf("node '%s' has %d incoming edge(s) but is labeled 'exogenous'; must be 'endogenous'",
                        nid, length(n_parents))
      )
    }
  }

  cycle_nodes <- detect_cycle_nodes(dag_nodes, dag_edges)
  if (length(cycle_nodes) > 0) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "acyclic",
      issue = sprintf("cycle detected involving nodes: %s",
                      paste(cycle_nodes, collapse = ", "))
    )
  }

  result <- list(
    valid = length(errors) == 0,
    n_variables = length(variables),
    errors = errors
  )

  jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
  cat(sprintf("Validation result written to %s\n", output_path))

  if (result$valid) {
    cat(sprintf("DAG is valid: %d nodes, %d edges, %d variables.\n",
                length(dag_nodes), length(dag_edges), length(variables)))
  } else {
    cat(sprintf("%d validation error(s) found:\n", length(errors)))
    for (e in errors) {
      cat(sprintf("  - [%s] %s: %s\n", e$variable, e$field, e$issue))
    }
  }

  invisible(result)
}

args <- commandArgs(trailingOnly = TRUE)
var_path <- if (length(args) >= 1) args[1] else "synthdata/variables.json"
dag_path <- if (length(args) >= 2) args[2] else "synthdata/dag.json"
out_path <- if (length(args) >= 3) args[3] else "synthdata/dag_validation_result.json"

validate_dag(var_path, dag_path, out_path)
