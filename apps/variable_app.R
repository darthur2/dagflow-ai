library(shiny)
library(jsonlite)

`%||%` <- function(a, b) if (is.null(a)) b else a

load_variables <- function(path = "synthdata/variables.json") {
  if (!file.exists(path)) {
    stop("variables.json not found")
  }
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

save_variables <- function(vars, path = "synthdata/variables.json") {
  json <- jsonlite::toJSON(vars, pretty = TRUE, auto_unbox = TRUE)
  writeLines(json, path)
}

format_category_names <- function(var) {
  paste(var$category_names %||% character(0), collapse = ", ")
}

parse_category_names <- function(str) {
  parts <- trimws(strsplit(str, ",")[[1]])
  parts[parts != ""]
}

ui <- fluidPage(
  titlePanel("Variable Editor"),
  sidebarLayout(
    sidebarPanel(
      width = 5,
      selectInput("variable_select", "Select Variable", choices = NULL),
      hr(),
      h4("Properties"),
      textInput("var_name", "Name"),
      textAreaInput("var_short_desc", "Short Description", rows = 2),
      textAreaInput("var_reason", "Reason for Inclusion", rows = 2),
      selectInput("var_data_type", "Data Type",
                  choices = c("quantitative", "categorical")),
      selectInput("var_meas_level", "Measurement Level",
                  choices = c("ratio", "interval", "ordinal", "nominal")),
      uiOutput("type_specific_fields"),
      br(),
      actionButton("save_btn", "Save Changes",
                   class = "btn-primary", style = "width: 100%;"),
      br(), br(),
      div(style = "display: flex; gap: 8px;",
          actionButton("add_btn", "Add Variable", style = "flex: 1;"),
          actionButton("delete_btn", "Delete Variable", style = "flex: 1;")
      )
    ),
    mainPanel(
      width = 7,
      h4("Variable Summary"),
      verbatimTextOutput("variable_summary")
    )
  )
)

server <- function(input, output, session) {
  values <- reactiveValues(vars = NULL)

  observe({
    values$vars <- load_variables()
  })

  observe({
    req(values$vars)
    names <- vapply(values$vars, function(v) v$name %||% "", character(1))
    choices <- setNames(seq_along(values$vars), names)
    selected <- isolate(input$variable_select)
    if (is.null(selected) || !selected %in% names(choices)) {
      selected <- choices[1]
    }
    updateSelectInput(session, "variable_select",
                      choices = choices, selected = selected)
  })

  observeEvent(input$variable_select, {
    req(values$vars, input$variable_select)
    idx <- as.integer(input$variable_select)
    var <- values$vars[[idx]]

    updateTextInput(session, "var_name", value = var$name %||% "")
    updateTextAreaInput(session, "var_short_desc",
                        value = var$short_description %||% "")
    updateTextAreaInput(session, "var_reason",
                        value = var$reason_for_inclusion %||% "")
    updateSelectInput(session, "var_data_type",
                      selected = var$data_type %||% "quantitative")
    updateSelectInput(session, "var_meas_level",
                      selected = var$measurement_level %||% "ratio")
  })

  current_var <- reactive({
    req(values$vars, input$variable_select)
    idx <- as.integer(input$variable_select)
    values$vars[[idx]]
  })

  output$type_specific_fields <- renderUI({
    req(current_var(), input$var_data_type)
    var <- current_var()
    dt <- input$var_data_type

    if (dt == "quantitative") {
      tagList(
        hr(),
        h5("Quantitative Fields"),
        selectInput("var_quant_type", "Quantitative Type",
                    choices = c("continuous", "discrete"),
                    selected = var$quantitative_type %||% "continuous"),
        selectInput("var_skew", "Skew",
                    choices = c("right", "left", "symmetric", "none"),
                    selected = var$skew %||% "none"),
        fluidRow(
          column(6, numericInput("var_min", "Min",
                                 value = var$bounds$min %||% 0)),
          column(6, numericInput("var_max", "Max",
                                 value = var$bounds$max %||% 1))
        )
      )
    } else {
      num_cat <- var$number_of_categories %||% 2
      cat_names <- format_category_names(var)
      tagList(
        hr(),
        h5("Categorical Fields"),
        numericInput("var_num_cat", "Number of Categories",
                     value = num_cat, min = 2, step = 1),
        textInput("var_cat_names", "Category Names (comma-separated)",
                  value = cat_names)
      )
    }
  })

  build_variable <- function() {
    req(input$variable_select)
    idx <- as.integer(input$variable_select)
    var <- values$vars[[idx]]

    var$name <- input$var_name
    var$short_description <- input$var_short_desc
    var$reason_for_inclusion <- input$var_reason
    var$data_type <- input$var_data_type
    var$measurement_level <- input$var_meas_level

    if (input$var_data_type == "quantitative") {
      var$quantitative_type <- input$var_quant_type
      var$bounds <- list(min = input$var_min, max = input$var_max)
      var$skew <- input$var_skew
      var$modality <- var$modality %||% "unimodal"
      var$number_of_categories <- NULL
      var$category_names <- NULL
    } else {
      cats <- parse_category_names(input$var_cat_names)
      var$number_of_categories <- length(cats)
      var$category_names <- as.list(cats)
      var$quantitative_type <- NULL
      var$bounds <- NULL
      var$skew <- NULL
      var$modality <- NULL
    }

    var
  }

  observeEvent(input$save_btn, {
    req(values$vars, input$variable_select)
    idx <- as.integer(input$variable_select)
    values$vars[[idx]] <- build_variable()
    save_variables(values$vars)
    showNotification("Saved to variables.json", type = "message", duration = 3)
  })

  observeEvent(input$add_btn, {
    new_var <- list(
      name = "new_variable",
      short_description = "",
      reason_for_inclusion = "",
      data_type = "quantitative",
      measurement_level = "ratio",
      effect_type = "fixed",
      quantitative_type = "continuous",
      bounds = list(min = 0, max = 1),
      skew = "none",
      modality = "unimodal"
    )
    values$vars <- c(values$vars, list(new_var))
    save_variables(values$vars)
    showNotification("New variable added", type = "message", duration = 3)
  })

  observeEvent(input$delete_btn, {
    req(values$vars, input$variable_select)
    idx <- as.integer(input$variable_select)
    var_name <- values$vars[[idx]]$name
    values$vars[[idx]] <- NULL
    if (length(values$vars) == 0) {
      values$vars <- list()
    }
    save_variables(values$vars)
    showNotification(paste0("Deleted: ", var_name), type = "message", duration = 3)
  })

  output$variable_summary <- renderText({
    req(current_var())
    var <- current_var()

    lines <- c(
      paste0("Name:                  ", var$name),
      paste0("Short Description:     ", var$short_description %||% ""),
      paste0("Reason for Inclusion:  ", var$reason_for_inclusion %||% ""),
      paste0("Data Type:             ", var$data_type),
      paste0("Measurement Level:     ", var$measurement_level),
      paste0("Effect Type:           ", var$effect_type %||% "fixed")
    )

    if (var$data_type == "quantitative") {
      lines <- c(lines,
        paste0("Quantitative Type:     ", var$quantitative_type %||% ""),
        paste0("Bounds:                [", var$bounds$min %||% "", ", ", var$bounds$max %||% "", "]"),
        paste0("Skew:                  ", var$skew %||% ""),
        paste0("Modality:              ", var$modality %||% "")
      )
    } else {
      cats <- format_category_names(var)
      lines <- c(lines,
        paste0("Number of Categories:  ", var$number_of_categories %||% ""),
        paste0("Category Names:        ", cats)
      )
    }

    paste(lines, collapse = "\n")
  })
}

shinyApp(ui, server)
