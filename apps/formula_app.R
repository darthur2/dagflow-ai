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
  if (length(predictors) == 0) {
    return(sprintf("%s ~ (no predictors)", lhs))
  }

  rhs_parts <- vapply(predictors, format_predictor_text, character(1))
  rhs <- paste(rhs_parts, collapse = "\n  + ")

  sprintf("%s ~ %.4f\n  + %s", lhs, as.numeric(eq$calibrated_beta0 %||% 0), rhs)
}

format_dist_summary <- function(eq) {
  dist <- eq$distribution
  params <- eq$distribution_parameters
  parts <- sprintf("%s: %s", names(params), unlist(params))
  paste(c(dist, parts), collapse = "\n")
}

format_coef_display <- function(coef) {
  if (length(coef) == 1) {
    sprintf("%.4f", as.numeric(coef))
  } else {
    paste(sprintf("%.4f", as.numeric(coef)), collapse = ", ")
  }
}

coef_input_id <- function(target, idx) {
  paste0("coef_", gsub("[^a-zA-Z0-9]", "_", target), "_", idx)
}

intercept_input_id <- function(target) {
  paste0("intercept_", gsub("[^a-zA-Z0-9]", "_", target))
}

ui <- fluidPage(
  titlePanel("Formula Explorer"),
  sidebarLayout(
    sidebarPanel(
      width = 4,
      h5("Load"),
      actionButton("load_btn", "Load from formulas.json",
                   class = "btn-primary", style = "width: 100%;"),
      br(), br(),
      selectInput("eq_select", "Select Target Variable", choices = NULL),
      hr(),
      h4("Response Details"),
      verbatimTextOutput("dist_summary"),
      hr(),
      h4("R²"),
      verbatimTextOutput("r2_display"),
      hr(),
      h4("Intercept"),
      verbatimTextOutput("intercept_display"),
      br(),
      actionButton("save_btn", "Save Changes",
                   class = "btn-warning", style = "width: 100%;")
    ),
    mainPanel(
      width = 8,
      h4("Formula"),
      verbatimTextOutput("formula_display"),
      hr(),
      h4("Predictors"),
      tableOutput("predictor_table"),
      hr(),
      h4("Calibrated Coefficients"),
      uiOutput("coef_editors")
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

  output$r2_display <- renderText({
    req(current_eq())
    r2 <- current_eq()$r2
    if (is.null(r2) || length(r2) == 0 || identical(r2, list())) {
      "N/A (categorical)"
    } else {
      sprintf("%.2f", as.numeric(r2))
    }
  })

  output$intercept_display <- renderText({
    req(current_eq())
    b0 <- current_eq()$calibrated_beta0
    if (is.null(b0)) {
      "Not calibrated"
    } else if (length(b0) == 1) {
      sprintf("%.4f", as.numeric(b0))
    } else {
      paste(sprintf("%.4f", as.numeric(b0)), collapse = ", ")
    }
  })

  output$formula_display <- renderText({
    req(current_eq())
    format_formula_string(current_eq())
  })

  output$predictor_table <- renderTable({
    req(current_eq())
    eq <- current_eq()
    preds <- eq$predictors
    if (length(preds) == 0) return(data.frame(Message = "No predictors"))
    do.call(rbind, lapply(preds, function(p) {
      col <- p$column
      ref <- p$reference %||% ""
      cats <- paste(unlist(p$categories %||% list()), collapse = ", ")
      coef_display <- format_coef_display(p$coefficient)
      data.frame(
        Predictor = col,
        Type = if (is.null(p$reference)) "continuous" else "categorical",
        Reference = ref,
        Categories = cats,
        Coefficients = coef_display,
        stringsAsFactors = FALSE
      )
    }))
  }, striped = TRUE, spacing = "s", width = "100%")

  output$coef_editors <- renderUI({
    req(current_eq())
    eq <- current_eq()
    target <- eq$target
    preds <- eq$predictors
    link <- distribution_link(eq$distribution)

    if (length(preds) == 0) return(p("No predictors to edit"))

    items <- list()
    idx <- 1
    for (p in preds) {
      col <- p$column
      coef <- as.numeric(p$coefficient)
      ref <- p$reference %||% ""
      cats <- unlist(p$categories %||% list())

      label <- if (length(cats) > 0) {
        sprintf("%s (ref=%s, cats=%s) [%s]", col, ref,
                paste(cats, collapse = ", "), link)
      } else {
        sprintf("%s [%s]", col, link)
      }

      if (length(coef) == 1) {
        items[[length(items) + 1]] <- numericInput(
          coef_input_id(target, idx),
          label = label,
          value = coef,
          step = 0.001
        )
        idx <- idx + 1
      } else {
        for (j in seq_along(coef)) {
          cat_label <- if (length(cats) >= j) sprintf("%s (%s)", label, cats[j]) else label
          items[[length(items) + 1]] <- numericInput(
            coef_input_id(target, idx),
            label = cat_label,
            value = coef[j],
            step = 0.001
          )
          idx <- idx + 1
        }
      }
    }

    b0 <- eq$calibrated_beta0 %||% 0
    b0_label <- sprintf("Intercept (calibrated_beta0) [%s]", link)
    if (length(b0) == 1) {
      items[[length(items) + 1]] <- numericInput(
        intercept_input_id(target),
        label = b0_label,
        value = as.numeric(b0),
        step = 0.001
      )
    } else {
      for (j in seq_along(b0)) {
        items[[length(items) + 1]] <- numericInput(
          paste0(intercept_input_id(target), "_", j),
          label = sprintf("%s [%d]", b0_label, j),
          value = as.numeric(b0[j]),
          step = 0.001
        )
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
        preds <- eq$predictors

        idx <- 1
        for (j in seq_along(preds)) {
          p <- preds[[j]]
          n_coef <- if (is.null(p$reference)) 1 else length(unlist(p$categories))
          if (n_coef == 1) {
            val <- input[[coef_input_id(target, idx)]]
            if (!is.null(val)) preds[[j]]$coefficient <- val
            idx <- idx + 1
          } else {
            new_coefs <- numeric(0)
            for (k in seq_len(n_coef)) {
              val <- input[[coef_input_id(target, idx)]]
              if (!is.null(val)) new_coefs <- c(new_coefs, val)
              idx <- idx + 1
            }
            if (length(new_coefs) == n_coef) preds[[j]]$coefficient <- new_coefs
          }
        }
        values$formulas$equations[[i]]$predictors <- preds

        b0_id <- intercept_input_id(target)
        b0 <- eq$calibrated_beta0
        if (length(b0) <= 1) {
          val <- input[[b0_id]]
          if (!is.null(val)) values$formulas$equations[[i]]$calibrated_beta0 <- val
        } else {
          new_b0 <- numeric(0)
          for (j in seq_along(b0)) {
            val <- input[[paste0(b0_id, "_", j)]]
            if (!is.null(val)) new_b0 <- c(new_b0, val)
          }
          if (length(new_b0) == length(b0)) {
            values$formulas$equations[[i]]$calibrated_beta0 <- new_b0
          }
        }

        break
      }
    }

    save_formulas(values$formulas)
    showNotification("Saved to formulas.json", type = "message", duration = 5)
  })
}

shinyApp(ui, server)
