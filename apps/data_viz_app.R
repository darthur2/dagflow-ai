library(shiny)
library(ggplot2)

find_project_root <- function() {
  d <- getwd()
  while (d != dirname(d)) {
    if (dir.exists(file.path(d, "synthdata"))) return(d)
    d <- dirname(d)
  }
  stop("Cannot find project root (synthdata/ not found)")
}

mosaic_data <- function(data, var1, var2) {
  tbl <- table(data[[var1]], data[[var2]])
  props <- prop.table(tbl, 1)
  df <- as.data.frame(props)
  colnames(df) <- c(var1, var2, "Proportion")

  margin <- prop.table(table(data[[var1]]))
  margin_df <- data.frame(
    Var1 = names(margin),
    Proportion_margin = as.numeric(margin)
  )
  colnames(margin_df)[1] <- var1

  df <- merge(df, margin_df, by = var1)
  df$xmin <- 0
  df$xmax <- 0
  df$ymin <- 0
  df$ymax <- 0

  for (v1 in unique(df[[var1]])) {
    idx <- df[[var1]] == v1
    total_w <- df$Proportion_margin[idx][1]
    cum_w <- 0
    for (v2 in unique(df[[var2]])) {
      sub_idx <- idx & df[[var2]] == v2
      if (any(sub_idx)) {
        h <- df$Proportion[sub_idx]
        df$xmin[sub_idx] <- cum_w
        df$xmax[sub_idx] <- cum_w + total_w * h
        cum_w <- cum_w + total_w * h
      }
    }
  }

  y_pos <- 0
  for (v1 in unique(df[[var1]])) {
    idx <- df[[var1]] == v1
    w <- df$Proportion_margin[idx][1]
    df$ymin[idx] <- y_pos
    df$ymax[idx] <- y_pos + w
    y_pos <- y_pos + w
  }

  df$x_center <- (df$xmin + df$xmax) / 2
  df$y_center <- (df$ymin + df$ymax) / 2

  df
}

ui <- fluidPage(
  titlePanel("Data Explorer"),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      h5("Load"),
      actionButton("load_btn", "Load from generated_data.csv",
                   class = "btn-primary", style = "width: 100%;"),
      br(), br(),
      selectInput("primary_var", "Primary Variable", choices = NULL),
      br(),
      uiOutput("secondary_controls")
    ),
    mainPanel(
      width = 9,
      uiOutput("tabs_ui")
    )
  )
)

server <- function(input, output, session) {
  values <- reactiveValues(data = NULL)

  observeEvent(input$load_btn, {
    path <- file.path(find_project_root(), "synthdata", "generated_data.csv")
    tryCatch({
      df <- read.csv(path, stringsAsFactors = FALSE)
      values$data <- df
      cols <- colnames(df)
      updateSelectInput(session, "primary_var", choices = cols, selected = cols[1])
      showNotification(paste("Loaded", nrow(df), "rows"), type = "message", duration = 3)
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error", duration = 10)
    })
  })

  is_quantitative <- reactive({
    req(values$data, input$primary_var)
    var <- input$primary_var
    is.numeric(values$data[[var]])
  })

  quant_vars <- reactive({
    req(values$data)
    cols <- colnames(values$data)
    cols[vapply(cols, function(x) is.numeric(values$data[[x]]), logical(1))]
  })

  cat_vars <- reactive({
    req(values$data)
    cols <- colnames(values$data)
    cols[!vapply(cols, function(x) is.numeric(values$data[[x]]), logical(1))]
  })

  output$secondary_controls <- renderUI({
    req(values$data, input$primary_var)

    if (is_quantitative()) {
      qv <- setdiff(quant_vars(), input$primary_var)
      cv <- cat_vars()
      tagList(
        h5("Scatterplot"),
        selectInput("scatter_x", "X Variable", choices = qv, selected = qv[1]),
        h5("Boxplot"),
        selectInput("box_group", "Group By", choices = cv, selected = cv[1])
      )
    } else {
      cv <- setdiff(cat_vars(), input$primary_var)
      tagList(
        h5("Mosaic Plot"),
        selectInput("mosaic_var2", "Second Variable", choices = cv, selected = cv[1])
      )
    }
  })

  output$tabs_ui <- renderUI({
    req(values$data, input$primary_var)

    if (is_quantitative()) {
      tabsetPanel(
        tabPanel("Histogram", plotOutput("hist_plot", height = "500px")),
        tabPanel("Scatterplot", plotOutput("scatter_plot", height = "500px")),
        tabPanel("Boxplot", plotOutput("box_plot", height = "500px"))
      )
    } else {
      tabsetPanel(
        tabPanel("Barplot", plotOutput("bar_plot", height = "500px")),
        tabPanel("Mosaic Plot", plotOutput("mosaic_plot", height = "500px"))
      )
    }
  })

  output$hist_plot <- renderPlot({
    req(values$data, input$primary_var)
    var <- input$primary_var
    ggplot(values$data, aes(x = .data[[var]])) +
      geom_histogram(fill = "#2c7fb8", color = "white", bins = 30) +
      labs(title = paste("Histogram of", var), x = var, y = "Count") +
      theme_minimal()
  })

  output$scatter_plot <- renderPlot({
    req(values$data, input$primary_var, input$scatter_x)
    xvar <- input$scatter_x
    yvar <- input$primary_var
    ggplot(values$data, aes(x = .data[[xvar]], y = .data[[yvar]])) +
      geom_point(alpha = 0.6, color = "#2c7fb8") +
      geom_smooth(method = "lm", se = TRUE, color = "#d95f02", fill = "#d95f02", alpha = 0.2) +
      labs(title = paste(yvar, "vs", xvar), x = xvar, y = yvar) +
      theme_minimal()
  })

  output$box_plot <- renderPlot({
    req(values$data, input$primary_var, input$box_group)
    yvar <- input$primary_var
    group <- input$box_group
    ggplot(values$data, aes(x = .data[[group]], y = .data[[yvar]], fill = .data[[group]])) +
      geom_boxplot(show.legend = FALSE) +
      scale_fill_viridis_d() +
      labs(title = paste(yvar, "by", group), x = group, y = yvar) +
      theme_minimal()
  })

  output$bar_plot <- renderPlot({
    req(values$data, input$primary_var)
    var <- input$primary_var
    ggplot(values$data, aes(x = .data[[var]])) +
      geom_bar(fill = "#2c7fb8", color = "white") +
      labs(title = paste("Barplot of", var), x = var, y = "Count") +
      theme_minimal()
  })

  output$mosaic_plot <- renderPlot({
    req(values$data, input$primary_var, input$mosaic_var2)
    var1 <- input$primary_var
    var2 <- input$mosaic_var2
    md <- mosaic_data(values$data, var1, var2)

    ggplot(md, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)) +
      geom_rect(aes(fill = .data[[var2]]), color = "white", linewidth = 0.5) +
      geom_text(aes(x = x_center, y = y_center,
                    label = sprintf("%.0f%%", Proportion * 100)),
                size = 3.5) +
      scale_fill_viridis_d() +
      labs(title = paste("Mosaic Plot:", var1, "vs", var2),
           x = var2, y = var1, fill = var2) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
}

shinyApp(ui, server)
