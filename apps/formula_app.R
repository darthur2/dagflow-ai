library(shiny)
library(jsonlite)

`%||%` <- function(a, b) if (is.null(a)) b else a

find_project_root <- function() {
  d <- getwd()
  while (d != dirname(d)) {
    if (dir.exists(file.path(d, "synthdata"))) return(d)
    d <- dirname(d)
  }
  stop("Cannot find project root (synthdata/ not found)")
}

load_formulas <- function(path = NULL) {
  if (is.null(path)) {
    path <- file.path(find_project_root(), "synthdata", "formulas.json")
  }
  if (!file.exists(path)) {
    stop("formulas.json not found")
  }
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

save_formulas <- function(formulas, path = NULL) {
  if (is.null(path)) {
    path <- file.path(find_project_root(), "synthdata", "formulas.json")
  }
  jsonlite::write_json(formulas, path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

distribution_link <- function(dist) {
  switch(dist,
    normal = "identity",
    gamma = "log",
    lognormal = "log",
    beta = "logit",
    poisson = "log",
    `negative binomial` = "log",
    binomial = "logit",
    `categorical-nominal` = "multinomial logit",
    `categorical-ordinal` = "cumulative logit",
    uniform = "probit",
    `discrete uniform` = "probit",
    "unknown"
  )
}

format_predictor_text <- function(p) {
  coef <- p$coefficient
  col <- p$column
  if (!is.null(p$reference)) {
    cats <- unlist(p$categories)
    coefs <- as.numeric(coef)
    parts <- sprintf("%s (%.3f)", cats, coefs)
    sprintf("%s [%s]", col, paste(parts, collapse = ", "))
  } else {
    sprintf("%s (%.4f)", col, as.numeric(coef))
  }
}

format_formula_string <- function(eq) {
  target <- eq$target
  dist <- eq$distribution

  lhs <- if (dist %in% c("lognormal", "gamma")) {
    sprintf("log(%s)", target)
  } else {
    target
  }

  predictors <- eq$predictors
  int <- eq$intercept %||% 0

  if (length(predictors) == 0) {
    return(sprintf("%s ~ %s", lhs, paste(int, collapse = ", ")))
  }

  rhs_parts <- vapply(predictors, format_predictor_text, character(1))
  rhs <- paste(rhs_parts, collapse = "\n  + ")

  sprintf("%s ~ %s\n  + %s", lhs, paste(int, collapse = ", "), rhs)
}

format_dist_summary <- function(eq) {
  dist <- eq$distribution
  params <- eq$distribution_parameters
  parts <- sprintf("%s: %s", names(params), unlist(params))
  paste(c(dist, parts), collapse = "\n")
}

input_id <- function(target, suffix) {
  paste0("f_", gsub("[^a-zA-Z0-9]", "_", target), "_", suffix)
}

ui <- fluidPage(
  titlePanel("Formula Explorer"),
  sidebarLayout(
    sidebarPanel(
      width = 5,
      h5("Load"),
      actionButton("load_btn", "Load from formulas.json",
                   class = "btn-primary", style = "width: 100%;"),
      br(), br(),
      selectInput("eq_select", "Select Target Variable", choices = NULL),
      hr(),
      h4("Distribution"),
      verbatimTextOutput("dist_summary"),
      hr(),
      h4("Parameters"),
      uiOutput("param_editors"),
      br(),
      actionButton("save_btn", "Save Changes",
                   class = "btn-warning", style = "width: 100%;")
    ),
    mainPanel(
      width = 7,
      h4("Formula"),
      verbatimTextOutput("formula_display")
    )
  )
)

server <- function(input, output, session) {
  values <- reactiveValues(formulas = NULL)

  load_from_file <- function(path) {
    tryCatch({
      formulas <- load_formulas(path)
      values$formulas <- formulas
      eq_names <- vapply(formulas$equations, `[[`, character(1), "target")
      updateSelectInput(session, "eq_select", choices = eq_names, selected = eq_names[1])
      showNotification(paste("Loaded", length(formulas$equations), "equations"),
                       type = "message", duration = 5)
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error", duration = 10)
    })
  }

  observeEvent(input$load_btn, {
    path <- file.path(find_project_root(), "synthdata", "formulas.json")
    load_from_file(path)
  })

  current_eq <- reactive({
    req(values$formulas, input$eq_select)
    target <- input$eq_select
    for (eq in values$formulas$equations) {
      if (eq$target == target) return(eq)
    }
    NULL
  })

  output$dist_summary <- renderText({
    req(current_eq())
    format_dist_summary(current_eq())
  })

  output$formula_display <- renderText({
    req(current_eq())
    format_formula_string(current_eq())
  })

  output$param_editors <- renderUI({
    req(current_eq())
    eq <- current_eq()
    target <- eq$target
    link <- distribution_link(eq$distribution)

    items <- list()

    r2 <- eq$r2
    if (is.null(r2) || length(r2) == 0 || identical(r2, list())) {
      items[[length(items) + 1]] <- p(style = "color: #666; font-size: 0.9em;",
                                       "R²: N/A (categorical target)")
    } else {
      items[[length(items) + 1]] <- numericInput(
        input_id(target, "r2"),
        label = "R² (0–1)",
        value = as.numeric(r2),
        min = 0, max = 1, step = 0.01
      )
    }

    b0 <- eq$intercept %||% 0
    if (length(b0) == 1) {
      items[[length(items) + 1]] <- numericInput(
        input_id(target, "int"),
        label = sprintf("Intercept [%s]", link),
        value = as.numeric(b0),
        step = 0.001
      )
    } else {
      items[[length(items) + 1]] <- h5("Intercept thresholds")
      for (j in seq_along(b0)) {
        items[[length(items) + 1]] <- numericInput(
          input_id(target, paste0("int_", j)),
          label = sprintf("Threshold %d [%s]", j, link),
          value = as.numeric(b0[j]),
          step = 0.001
        )
      }
    }

    preds <- eq$predictors
    if (length(preds) > 0) {
      items[[length(items) + 1]] <- hr()
      items[[length(items) + 1]] <- h4("Coefficients")
      idx <- 1
      for (p in preds) {
        coef <- as.numeric(p$coefficient)
        cats <- unlist(p$categories %||% list())

        if (length(cats) > 0) {
          items[[length(items) + 1]] <- h5(sprintf("%s (ref=%s)", p$column, p$reference %||% ""))
          for (j in seq_along(coef)) {
            items[[length(items) + 1]] <- numericInput(
              input_id(target, paste0("coef_", idx)),
              label = sprintf("  %s [%s]", cats[j], link),
              value = coef[j],
              step = 0.001
            )
            idx <- idx + 1
          }
        } else {
          items[[length(items) + 1]] <- numericInput(
            input_id(target, paste0("coef_", idx)),
            label = sprintf("%s [%s]", p$column, link),
            value = coef,
            step = 0.001
          )
          idx <- idx + 1
        }
      }
    }

    do.call(tagList, items)
  })

  observeEvent(input$save_btn, {
    req(values$formulas, input$eq_select)
    target <- input$eq_select

    for (i in seq_along(values$formulas$equations)) {
      if (values$formulas$equations[[i]]$target == target) {
        eq <- values$formulas$equations[[i]]

        r2_val <- input[[input_id(target, "r2")]]
        if (!is.null(r2_val)) {
          values$formulas$equations[[i]]$r2 <- r2_val
        }

        b0 <- eq$intercept %||% 0
        if (length(b0) <= 1) {
          val <- input[[input_id(target, "int")]]
          if (!is.null(val)) values$formulas$equations[[i]]$intercept <- val
        } else {
          new_b0 <- numeric(0)
          for (j in seq_along(b0)) {
            val <- input[[input_id(target, paste0("int_", j))]]
            if (!is.null(val)) new_b0 <- c(new_b0, val)
          }
          if (length(new_b0) == length(b0)) {
            values$formulas$equations[[i]]$intercept <- new_b0
          }
        }

        preds <- eq$predictors
        if (length(preds) > 0) {
          idx <- 1
          for (j in seq_along(preds)) {
            p <- preds[[j]]
            n_coef <- if (is.null(p$reference)) 1 else length(unlist(p$categories))
            if (n_coef == 1) {
              val <- input[[input_id(target, paste0("coef_", idx))]]
              if (!is.null(val)) preds[[j]]$coefficient <- val
              idx <- idx + 1
            } else {
              new_coefs <- numeric(0)
              for (k in seq_len(n_coef)) {
                val <- input[[input_id(target, paste0("coef_", idx))]]
                if (!is.null(val)) new_coefs <- c(new_coefs, val)
                idx <- idx + 1
              }
              if (length(new_coefs) == n_coef) preds[[j]]$coefficient <- new_coefs
            }
          }
          values$formulas$equations[[i]]$predictors <- preds
        }

        break
      }
    }

    save_formulas(values$formulas)
    showNotification("Saved to formulas.json", type = "message", duration = 5)
  })
}

shinyApp(ui, server)
