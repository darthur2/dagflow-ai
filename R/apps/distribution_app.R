library(shiny)
library(ggplot2)
library(jsonlite)

find_project_root <- function() {
  d <- getwd()
  while (d != dirname(d)) {
    if (dir.exists(file.path(d, "synthdata")) || dir.exists(file.path(d, "R"))) return(d)
    d <- dirname(d)
  }
  getwd()
}

root <- find_project_root()
source(file.path(root, "R", "plot_distribution.R"))

get_param_names <- function(distribution) {
  switch(distribution,
    normal = c("mean", "sd", "min", "max"),
    gamma = c("shape", "rate", "min", "max"),
    beta = c("shape1", "shape2", "min", "max"),
    lognormal = c("meanlog", "sdlog", "min", "max"),
    uniform = c("min", "max"),
    `discrete uniform` = c("min", "max"),
    binomial = c("size", "prob", "min", "max"),
    `negative binomial` = c("size", "mu", "min", "max"),
    poisson = c("lambda", "min", "max"),
    `categorical-nominal` = c("categories", "probabilities"),
    `categorical-ordinal` = c("categories", "probabilities"),
  )
}

get_step_size <- function(param_name) {
  if (param_name %in% c("size")) 1 else 0.01
}

load_variable_json <- function(json_str) {
  parsed <- jsonlite::fromJSON(json_str, simplifyVector = FALSE)
  if (!is.list(parsed) || length(parsed) == 0) {
    stop("JSON must be a non-empty array")
  }
  for (i in seq_along(parsed)) {
    if (is.null(parsed[[i]]$name)) stop("Variable ", i, " missing 'name'")
    if (is.null(parsed[[i]]$distribution)) {
      stop("Variable ", i, " missing 'distribution'")
    }
    if (is.null(parsed[[i]]$distribution_parameters)) {
      stop("Variable ", i, " missing 'distribution_parameters'")
    }
  }
  parsed
}

default_json <- jsonlite::toJSON(
  list(
    list(name = "Click \"Load from distributions.json\"",
         distribution = "uniform",
         distribution_parameters = list(min = 0, max = 1))
  ),
  pretty = TRUE, auto_unbox = TRUE
)

ui <- fluidPage(
  titlePanel("Distribution Explorer"),
  sidebarLayout(
    sidebarPanel(
      width = 4,

      h5("Load from file"),
      actionButton("load_from_file", "Load from distributions.json",
        class = "btn-primary", style = "width: 100%;"),
      br(), br(),
      fileInput("json_file_upload", NULL,
        buttonLabel = "Upload JSON File",
        accept = ".json"),
      hr(),

      h5("Or paste JSON"),
      textAreaInput("json_input", label = NULL,
        value = default_json, rows = 4, resize = "vertical"),
      actionButton("load_json", "Load from Text",
        style = "width: 100%;"),
      hr(),

      selectInput("variable_name", "Select Variable", choices = NULL),
      hr(),

      h4("Parameters"),
      uiOutput("param_controls")
    ),
    mainPanel(
      width = 8,
      plotOutput("distribution_plot", height = "500px"),
      hr(),
      div(
        style = "display: flex; justify-content: space-between; align-items: center;",
        h4("Updated JSON Output", style = "margin: 0;"),
        div(
          downloadButton("download_json", "Download JSON"),
          actionButton("save_refined", "Save Refined JSON",
            class = "btn-warning", style = "margin-left: 8px;")
        )
      ),
      verbatimTextOutput("json_output")
    )
  )
)

server <- function(input, output, session) {
  values <- reactiveValues(vars = NULL)

  populate_vars <- function(parsed) {
    values$vars <- parsed
    names <- vapply(parsed, function(v) v$name, character(1))
    updateSelectInput(session, "variable_name", choices = names, selected = names[1])
  }

  load_from_json_str <- function(json_str, source_label = "input") {
    tryCatch({
      parsed <- load_variable_json(json_str)
      populate_vars(parsed)
    }, error = function(e) {
      showNotification(paste("Error in", source_label, ":", e$message),
        type = "error", duration = 10)
    })
  }

  load_from_file <- function(path, source_label = "file") {
    tryCatch({
      if (!file.exists(path)) {
        stop("File not found: ", path)
      }
      json_str <- paste(readLines(path, warn = FALSE), collapse = "\n")
      parsed <- load_variable_json(json_str)
      populate_vars(parsed)
      showNotification(paste("Loaded", length(parsed), "variables from", path),
        type = "message", duration = 5)
    }, error = function(e) {
      showNotification(paste("Error loading", source_label, ":", e$message),
        type = "error", duration = 10)
    })
  }

  observeEvent(input$load_from_file, {
    path <- file.path(find_project_root(), "synthdata", "distributions.json")
    load_from_file(path, "distributions.json")
  })

  observeEvent(input$json_file_upload, {
    req(input$json_file_upload$datapath)
    load_from_file(input$json_file_upload$datapath, "uploaded file")
  })

  observeEvent(input$load_json, {
    req(input$json_input)
    load_from_json_str(input$json_input, "text input")
  })

  current_var_index <- reactive({
    req(values$vars, input$variable_name)
    which(vapply(values$vars, function(v) v$name == input$variable_name,
      logical(1)))[1]
  })

  current_distribution <- reactive({
    req(current_var_index())
    values$vars[[current_var_index()]]$distribution
  })

  current_params <- reactive({
    req(current_var_index())
    values$vars[[current_var_index()]]$distribution_parameters
  })

  output$param_controls <- renderUI({
    req(current_distribution(), current_params())
    dist <- current_distribution()
    params <- current_params()

    if (dist %in% c("categorical-nominal", "categorical-ordinal")) {
      cats <- unlist(params$categories)
      probs <- unlist(params$probabilities)
      tagList(
        lapply(seq_along(cats), function(i) {
          sliderInput(paste0("prob_", i),
            label = cats[i],
            min = 0, max = 1, value = probs[i], step = 0.01)
        }),
        br(),
        strong("Normalized probabilities:"),
        verbatimTextOutput("prob_norm_display")
      )
    } else {
      param_names <- get_param_names(dist)
      tagList(
        lapply(param_names, function(pname) {
          value <- params[[pname]]
          if (is.null(value)) value <- 0
          numericInput(paste0("param_", pname),
            label = pname,
            value = value,
            step = get_step_size(pname))
        })
      )
    }
  })

  current_param_values <- reactive({
    req(values$vars, current_var_index(), current_distribution())
    dist <- current_distribution()
    idx <- current_var_index()
    params <- values$vars[[idx]]$distribution_parameters

    if (dist %in% c("categorical-nominal", "categorical-ordinal")) {
      cats <- unlist(params$categories)
      raw_probs <- vapply(seq_along(cats), function(i) {
        val <- input[[paste0("prob_", i)]]
        if (is.null(val)) params$probabilities[[i]] else val
      }, numeric(1))
      if (all(raw_probs == 0)) raw_probs <- rep(1, length(raw_probs))
      norm_probs <- raw_probs / sum(raw_probs)
      list(categories = cats, probabilities = norm_probs)
    } else {
      param_names <- setdiff(get_param_names(dist), c("categories", "probabilities"))
      result <- list()
      for (pname in param_names) {
        val <- input[[paste0("param_", pname)]]
        if (is.null(val) || !is.numeric(val) || length(val) != 1) {
          result[[pname]] <- params[[pname]]
        } else {
          result[[pname]] <- val
        }
      }
      result
    }
  })

  output$prob_norm_display <- renderPrint({
    req(current_distribution())
    dist <- current_distribution()
    req(dist %in% c("categorical-nominal", "categorical-ordinal"))
    pv <- current_param_values()
    probs <- pv$probabilities
    names(probs) <- pv$categories
    print(probs)
  })

  output$distribution_plot <- renderPlot({
    req(values$vars, current_var_index())
    dist <- current_distribution()
    params <- current_param_values()

    plot_fn <- switch(dist,
      normal = plot_normal_marginal,
      gamma = plot_gamma_distribution,
      beta = plot_beta_distribution,
      lognormal = plot_lognormal_distribution,
      uniform = plot_uniform_distribution,
      `discrete uniform` = plot_discrete_uniform_distribution,
      binomial = plot_binomial_distribution,
      `negative binomial` = plot_negative_binomial_distribution,
      poisson = plot_poisson_distribution,
      `categorical-nominal` = plot_categorical_nominal_distribution,
      `categorical-ordinal` = plot_categorical_ordinal_distribution
    )

    tryCatch({
      do.call(plot_fn, params)
    }, error = function(e) {
      ggplot2::ggplot() +
        ggplot2::annotate("text", x = 0.5, y = 0.5,
          label = paste("Plot error:", e$message),
          size = 5, hjust = 0.5) +
        ggplot2::theme_void()
    })
  })

  current_json <- reactive({
    req(values$vars)
    result <- lapply(seq_along(values$vars), function(i) {
      v <- values$vars[[i]]
      if (i == current_var_index()) {
        params <- current_param_values()
      } else {
        params <- v$distribution_parameters
      }
      list(
        name = v$name,
        distribution = v$distribution,
        distribution_parameters = params
      )
    })
    jsonlite::toJSON(result, pretty = TRUE, auto_unbox = TRUE)
  })

  output$json_output <- renderText({
    current_json()
  })

  output$download_json <- downloadHandler(
    filename = function() {
      "distributions.json"
    },
    content = function(file) {
      writeLines(current_json(), file)
    }
  )

  observeEvent(input$save_refined, {
    req(values$vars)
    path <- file.path(find_project_root(), "synthdata", "distributions.json")
    writeLines(current_json(), path)
    showNotification(
      "Saved to synthdata/distributions.json",
      type = "message", duration = 5)
  })
}

shinyApp(ui, server)
