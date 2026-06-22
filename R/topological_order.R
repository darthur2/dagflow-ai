topological_sort <- function(nodes, edges) {
  node_ids <- vapply(nodes, `[[`, character(1), "id")
  types <- vapply(nodes, `[[`, character(1), "type")
  node_types <- setNames(types, node_ids)

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

  all_zero <- node_ids[in_degree == 0L]
  is_exo <- node_types[all_zero] == "exogenous"
  exo_first <- sort(all_zero[is_exo])
  endo_rest <- sort(all_zero[!is_exo])
  queue <- c(exo_first, endo_rest)

  order <- character(0)

  while (length(queue) > 0) {
    current <- queue[1]
    queue <- queue[-1]
    order <- c(order, current)

    for (neighbor in adj[[current]]) {
      in_degree[neighbor] <- in_degree[neighbor] - 1L
      if (in_degree[neighbor] == 0L) {
        queue <- c(queue, neighbor)
      }
    }
  }

  if (length(order) != length(node_ids)) {
    stop("DAG contains a cycle — topological sort not possible")
  }

  order
}
