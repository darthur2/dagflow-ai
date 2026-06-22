library(shiny)
library(visNetwork)
library(jsonlite)

find_project_root <- function() {
  d <- getwd()
  while (d != dirname(d)) {
    if (dir.exists(file.path(d, "synthdata"))) return(d)
    d <- dirname(d)
  }
  stop("Cannot find project root (synthdata/ not found)")
}

root <- find_project_root()
source(file.path(root, "R", "plot_dag.R"))

load_dag_file <- function(path) {
  if (!file.exists(path)) {
    stop("File not found: ", path)
  }
  json_str <- paste(readLines(path, warn = FALSE), collapse = "\n")
  parsed <- jsonlite::fromJSON(json_str, simplifyVector = FALSE)
  if (is.null(parsed$nodes) || is.null(parsed$edges)) {
    stop("DAG JSON must have 'nodes' and 'edges' fields")
  }
  parsed
}

get_node_ids <- function(dag) {
  vapply(dag$nodes, `[[`, character(1), "id")
}

get_type <- function(node) {
  if (is.null(node$type)) "endogenous" else node$type
}

edge_label <- function(from, to) {
  paste0(from, " -> ", to)
}

ui <- fluidPage(
  titlePanel("DAG Explorer"),
  sidebarLayout(
    sidebarPanel(
      width = 4,

      h5("Load"),
      actionButton("load_from_file", "Load from dag.json",
        class = "btn-primary", style = "width: 100%;"),
      br(), br(),

      h5("Exogenous Variables"),
      p("Check to include, uncheck to hide.",
        style = "font-size: 12px; color: #666;"),
      uiOutput("exogenous_checkboxes"),
      hr(),

      h5("Add Edge"),
      selectInput("edge_from", "From", choices = NULL),
      selectInput("edge_to", "To", choices = NULL),
      actionButton("add_edge", "Add Edge",
        style = "width: 100%;"),
      hr(),

      h5("Remove Edge"),
      selectInput("remove_edge_select", "Select Edge", choices = NULL),
      actionButton("remove_edge", "Remove Edge",
        style = "width: 100%;"),
      hr(),

      h5("Add Node"),
      textInput("new_node_name", "Name", placeholder = "variable_name"),
      selectInput("new_node_type", "Type",
        choices = c("endogenous", "exogenous")),
      actionButton("add_node", "Add Node",
        style = "width: 100%;"),
      hr(),

      h5("Remove Node"),
      selectInput("remove_node_select", "Select Node", choices = NULL),
      actionButton("remove_node", "Remove Node",
        style = "width: 100%;"),
      hr(),

      actionButton("save_dag", "Save DAG",
        class = "btn-warning", style = "width: 100%;")
    ),
    mainPanel(
      width = 8,
      visNetworkOutput("dag_plot", height = "550px"),
      hr(),
      h4("DAG JSON Output"),
      verbatimTextOutput("dag_json_output")
    )
  )
)

server <- function(input, output, session) {
  values <- reactiveValues(
    dag = NULL,
    all_exogenous = list(),
    exo_trigger = 0
  )

  available_exogenous <- reactive({
    values$all_exogenous
  })

  active_exogenous <- reactive({
    exo <- available_exogenous()
    if (length(exo) == 0) return(character(0))
    ids <- character(0)
    for (n in exo) {
      cb_id <- paste0("exo_", n$id)
      val <- input[[cb_id]]
      if (isTRUE(val)) ids <- c(ids, n$id)
    }
    ids
  })

  get_active_nodes <- reactive({
    req(values$dag)
    active_exo <- active_exogenous()
    Filter(function(n) {
      get_type(n) == "endogenous" || n$id %in% active_exo
    }, values$dag$nodes)
  })

  get_active_edges <- reactive({
    req(values$dag)
    active_nodes <- vapply(get_active_nodes(), `[[`, character(1), "id")
    Filter(function(e) {
      e$from %in% active_nodes && e$to %in% active_nodes
    }, values$dag$edges)
  })

  populate_app <- function(dag) {
    values$dag <- dag
    values$all_exogenous <- Filter(function(n) get_type(n) == "exogenous", dag$nodes)
    values$exo_trigger <- values$exo_trigger + 1
    update_ui_choices()
  }

  update_ui_choices <- function() {
    req(values$dag)
    active_nodes <- vapply(get_active_nodes(), `[[`, character(1), "id")

    updateSelectInput(session, "edge_from", choices = active_nodes)
    updateSelectInput(session, "edge_to", choices = active_nodes)
    updateSelectInput(session, "remove_node_select", choices = active_nodes)

    edges <- values$dag$edges
    if (length(edges) > 0) {
      all_active <- active_nodes
      filtered_edges <- Filter(function(e) {
        e$from %in% all_active && e$to %in% all_active
      }, edges)
      if (length(filtered_edges) > 0) {
        edge_labels <- vapply(filtered_edges, function(e) {
          edge_label(e$from, e$to)
        }, character(1))
        updateSelectInput(session, "remove_edge_select", choices = edge_labels)
      } else {
        updateSelectInput(session, "remove_edge_select", choices = character(0))
      }
    } else {
      updateSelectInput(session, "remove_edge_select", choices = character(0))
    }
  }

  observeEvent(input$load_from_file, {
    tryCatch({
      dag <- load_dag_file(file.path(find_project_root(), "synthdata", "dag.json"))
      populate_app(dag)
      showNotification(
        paste("Loaded", length(dag$nodes), "nodes and",
          length(dag$edges), "edges from dag.json"),
        type = "message", duration = 5)
    }, error = function(e) {
      showNotification(paste("Error:", e$message),
        type = "error", duration = 10)
    })
  })

  output$exogenous_checkboxes <- renderUI({
    req(values$dag)
    exo <- isolate(available_exogenous())
    if (length(exo) == 0) {
      return(p("No exogenous variables proposed."))
    }

    items <- lapply(exo, function(n) {
      desc <- if (!is.null(n$description)) n$description else ""
      cb_id <- paste0("exo_", n$id)
      current_val <- isolate({
        val <- input[[cb_id]]
        if (is.null(val)) TRUE else isTRUE(val)
      })
      checkboxInput(
        inputId = cb_id,
        label = HTML(paste0("<strong>", n$id, "</strong>",
          if (desc != "") paste0("<br><small>", desc, "</small>"))),
        value = current_val
      )
    })
    do.call(tagList, items)
  }) |>
    bindEvent(values$exo_trigger)

  observe({
    req(values$dag)
    active_exogenous()
    update_ui_choices()
  })

  observeEvent(input$add_edge, {
    req(values$dag)
    from <- input$edge_from
    to <- input$edge_to
    if (is.null(from) || is.null(to) || from == "" || to == "") {
      showNotification("Select both 'From' and 'To' nodes",
        type = "warning", duration = 5)
      return()
    }
    if (from == to) {
      showNotification("Cannot create a self-loop (acyclic graph required)",
        type = "warning", duration = 5)
      return()
    }

    edges <- values$dag$edges
    already_exists <- any(vapply(edges, function(e) {
      e$from == from && e$to == to
    }, logical(1)))

    if (already_exists) {
      showNotification("This edge already exists",
        type = "warning", duration = 5)
      return()
    }

    new_edge <- list(from = from, to = to)
    edges[[length(edges) + 1]] <- new_edge
    values$dag$edges <- edges
    update_ui_choices()
  })

  observeEvent(input$remove_edge, {
    req(values$dag)
    selected <- input$remove_edge_select
    if (is.null(selected) || selected == "") {
      showNotification("Select an edge to remove",
        type = "warning", duration = 5)
      return()
    }

    parts <- strsplit(selected, " -> ")[[1]]
    if (length(parts) != 2) return()

    edges <- values$dag$edges
    edges <- Filter(function(e) {
      !(e$from == parts[1] && e$to == parts[2])
    }, edges)
    values$dag$edges <- edges
    update_ui_choices()
  })

  observeEvent(input$add_node, {
    req(values$dag)
    name <- trimws(input$new_node_name)
    if (name == "") {
      showNotification("Enter a node name",
        type = "warning", duration = 5)
      return()
    }

    all_ids <- get_node_ids(values$dag)
    if (name %in% all_ids) {
      showNotification("A node with this name already exists",
        type = "warning", duration = 5)
      return()
    }

    ntype <- input$new_node_type
    new_node <- list(id = name, type = ntype)
    nodes <- values$dag$nodes
    nodes[[length(nodes) + 1]] <- new_node
    values$dag$nodes <- nodes

    if (ntype == "exogenous") {
      all_exo <- values$all_exogenous
      all_exo[[length(all_exo) + 1]] <- new_node
      values$all_exogenous <- all_exo
      values$exo_trigger <- values$exo_trigger + 1
    }

    updateTextInput(session, "new_node_name", value = "")
    update_ui_choices()
  })

  observeEvent(input$remove_node, {
    req(values$dag)
    selected <- input$remove_node_select
    if (is.null(selected) || selected == "") {
      showNotification("Select a node to remove",
        type = "warning", duration = 5)
      return()
    }

    nodes <- values$dag$nodes
    nodes <- Filter(function(n) n$id != selected, nodes)
    values$dag$nodes <- nodes

    edges <- values$dag$edges
    edges <- Filter(function(e) {
      e$from != selected && e$to != selected
    }, edges)
    values$dag$edges <- edges

    all_exo <- Filter(function(n) n$id != selected, values$all_exogenous)
    values$all_exogenous <- all_exo
    if (length(all_exo) != length(values$all_exogenous)) {
      values$exo_trigger <- values$exo_trigger + 1
    }
    values$all_exogenous <- all_exo
    values$exo_trigger <- values$exo_trigger + 1

    update_ui_choices()
  })

  output$dag_plot <- renderVisNetwork({
    req(values$dag)
    active_nodes <- get_active_nodes()
    active_edges <- get_active_edges()

    if (length(active_nodes) == 0) {
      return(NULL)
    }

    plot_dag_visnetwork(active_nodes, active_edges)
  })

  current_json <- reactive({
    req(values$dag)
    jsonlite::toJSON(
      list(
        nodes = values$dag$nodes,
        edges = values$dag$edges
      ),
      pretty = TRUE, auto_unbox = TRUE
    )
  })

  output$dag_json_output <- renderText({
    req(values$dag)
    current_json()
  })

  observeEvent(input$save_dag, {
    req(values$dag)
    writeLines(current_json(), file.path(find_project_root(), "synthdata", "dag.json"))
    showNotification("Saved to synthdata/dag.json",
      type = "message", duration = 5)
  })
}

shinyApp(ui, server)
