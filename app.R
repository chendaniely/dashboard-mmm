# Application note:
# This application is not written in the most efficient way,
# it will make a system call to re-fit models
library(tidyverse)
library(shiny)
library(bslib)
library(plotly)

# Load and prepare data (same as model.R lines 104-110)
data <- read_csv("data/synthetic-marketing-sales.csv")

data_model <- data |>
  select(date, week, year, quarter, month, week_of_year, holiday_week, sales_revenue,
    ends_with("_spend"),
    -total_marketing_spend
  ) |>
  mutate(
    date = as.Date(date),
    holiday_week = factor(holiday_week, levels = c(0, 1), labels = c("No", "Yes"))
  )

# Load models and params
model1 <- readRDS('saved_models/model1_simple_lr.rds')
model2 <- readRDS('saved_models/model2_transformed_lr.rds')
model3 <- readRDS('saved_models/model3_bayes_transformed.rds')
model4 <- readRDS('saved_models/model4_bayes_raw.rds')
params <- readRDS('saved_models/transformation_params.rds')

# Get channel information
spend_cols <- names(data_model)[str_detect(names(data_model), "_spend$")]
channel_names <- spend_cols |>
  str_remove("_spend") |>
  str_replace_all("_", " ") |>
  str_to_title()
names(spend_cols) <- channel_names

# UI
ui <- page_navbar(
  title = "Marketing Mix Modeling Dashboard - BrickCraft Studios",
  theme = bs_theme(version = 5),

  nav_panel(
    "Overview",
    layout_sidebar(
      sidebar = sidebar(
        width = 300,
        h4("Dashboard Controls"),
        dateRangeInput(
          "date_range",
          "Date Range:",
          start = min(data_model$date),
          end = max(data_model$date)
        ),
        checkboxGroupInput(
          "channels",
          "Select Channels:",
          choices = channel_names,
          selected = channel_names
        ),
        hr(),
        markdown(
          "**About This Dashboard**

          *This project contains synthetic data and analysis created for demonstration purposes only.*

          Explore marketing performance, channel contributions, and budget optimization insights.
          "
        )
      ),

      card(
        uiOutput("kpi_cards"),
      ),

      card(
        card_header("Sales & Marketing Spend Over Time"),
        plotlyOutput("sales_trend_plot", height = "200px")
      ),

      layout_columns(
        card(
          card_header("Channel Spend Distribution"),
          plotlyOutput("channel_distribution_plot", height = "150px")
        ),
        card(
          card_header("ROI by Channel"),
          plotlyOutput("roi_chart", height = "150px")
        )
      )
    )
  ),

  nav_panel(
    "Channel Analysis",
    layout_sidebar(
      sidebar = sidebar(
        width = 300,
        h4("Channel Selection"),
        selectInput(
          "analysis_channel",
          "Select Channel:",
          choices = channel_names,
          selected = channel_names[1]
        ),
        hr(),
        h4("Channel Statistics"),
        uiOutput("channel_stats")
      ),
      card(
        card_header("Spend vs Sales Contribution"),
        plotlyOutput("channel_scatter_plot", height = "400px")
      ),
      card(
        card_header("Weekly Performance"),
        plotlyOutput("channel_time_series", height = "400px")
      )
    )
  ),

  nav_panel(
    "Budget Optimizer",
    layout_sidebar(
      sidebar = sidebar(
        width = 300,
        h4("Budget Optimization"),
        numericInput(
          "total_budget",
          "Total Weekly Budget ($K):",
          value = round(mean(rowSums(data_model[, spend_cols]))),
          min = 50,
          max = 500,
          step = 10
        ),
        sliderInput(
          "min_allocation",
          "Min Channel Allocation (%):",
          min = 0,
          max = 20,
          value = 5,
          step = 1
        ),
        sliderInput(
          "max_allocation",
          "Max Channel Allocation (%):",
          min = 20,
          max = 100,
          value = 40,
          step = 5
        ),
        actionButton(
          "optimize",
          "Optimize Budget",
          class_ = "btn-primary"
        )
      ),
      card(
        card_header("Recommended Budget Allocation"),
        uiOutput("optimization_results")
      ),
      card(
        card_header("Current vs Recommended Allocation"),
        plotlyOutput("allocation_comparison", height = "400px")
      )
    )
  ),

nav_panel(
    "Model Configuration",
    layout_sidebar(
      sidebar = sidebar(
        width = 300,
        h4("Model Selection"),
        selectInput(
          "selected_model",
          "Choose Model:",
          choices = c(
            "Simple Linear Regression" = "model1",
            "Transformed Linear Regression" = "model2",
            "Bayesian (Transformed)" = "model3",
            "Bayesian (Raw)" = "model4"
          ),
          selected = "model2"
        ),
        hr(),
        h4("Model Performance"),
        uiOutput("model_performance")
      ),
      card(
        card_header("Model Information & Parameters"),
        uiOutput("model_info"),
        conditionalPanel(
          condition = "input.selected_model == 'model2' || input.selected_model == 'model3'",
          hr(),
          h4("Transformation Parameters"),
          layout_columns(
            div(
              sliderInput(
                "decay_rate",
                "Adstock Decay Rate:",
                min = 0,
                max = 1,
                value = params$decay_rate,
                step = 0.05
              ),
              helpText("How much marketing effect carries over week-to-week (0-1)")
            ),
            div(
              numericInput(
                "alpha",
                "Saturation Alpha (Scale):",
                value = params$alpha,
                min = 0.1,
                max = 5,
                step = 0.1
              ),
              helpText("Maximum possible effect (scaling factor)")
            ),
            div(
              sliderInput(
                "gamma",
                "Saturation Gamma (Shape):",
                min = 0.1,
                max = 1.5,
                value = params$gamma,
                step = 0.05
              ),
              helpText("Shape of diminishing returns curve")
            )
          ),
          actionButton(
            "retrain_models",
            "Retrain Models with New Parameters",
            class_ = "btn-primary"
          ),
          br(),
          textOutput("retrain_status")
        )
      )
    )
  )

)

# Server
server <- function(input, output, session) {

  # Reactive: filtered data based on date range (with contributions)
  filtered_data <- reactive({
    data_with_contributions() |>
      filter(
        date >= input$date_range[1],
        date <= input$date_range[2]
      )
  })

  # Reactive value to track model reloading
  models_version <- reactiveVal(0)

  # Reactive: Load models (reloads when models_version changes)
  current_models <- reactive({
    models_version()  # Depend on this to trigger reload

    list(
      model1 = readRDS('saved_models/model1_simple_lr.rds'),
      model2 = readRDS('saved_models/model2_transformed_lr.rds'),
      model3 = readRDS('saved_models/model3_bayes_transformed.rds'),
      model4 = readRDS('saved_models/model4_bayes_raw.rds'),
      params = readRDS('saved_models/transformation_params.rds')
    )
  })

  # Reactive: Data with pre-computed contributions for all models
  data_with_contributions <- reactive({
    models <- current_models()
    df <- data_model

    # Model 1: Simple Linear Regression
    coefs1 <- models$model1$coefficients |> filter(str_detect(term, "_spend"))
    for (spend_col in spend_cols) {
      channel <- str_remove(spend_col, "_spend")
      coef_row <- coefs1 |> filter(str_detect(term, channel))
      if (nrow(coef_row) > 0) {
        coef_val <- coef_row$estimate[1]
        df[[paste0(spend_col, "_contrib_model1")]] <- df[[spend_col]] * coef_val
      } else {
        df[[paste0(spend_col, "_contrib_model1")]] <- 0
      }
    }

    # Model 2: Transformed Linear Regression
    coefs2 <- models$model2$coefficients |> filter(str_detect(term, "_transformed"))
    for (spend_col in spend_cols) {
      channel <- str_remove(spend_col, "_spend")
      coef_row <- coefs2 |> filter(str_detect(term, channel))
      if (nrow(coef_row) > 0) {
        coef_val <- coef_row$estimate[1]
        df[[paste0(spend_col, "_contrib_model2")]] <- df[[spend_col]] * coef_val
      } else {
        df[[paste0(spend_col, "_contrib_model2")]] <- 0
      }
    }

    # Model 3: Bayesian Transformed
    coefs3 <- models$model3$coefficients |> filter(str_detect(variable, "_transformed"))
    for (spend_col in spend_cols) {
      channel <- str_remove(spend_col, "_spend")
      coef_row <- coefs3 |> filter(str_detect(variable, channel))
      if (nrow(coef_row) > 0) {
        coef_val <- coef_row$mean[1]
        df[[paste0(spend_col, "_contrib_model3")]] <- df[[spend_col]] * coef_val
      } else {
        df[[paste0(spend_col, "_contrib_model3")]] <- 0
      }
    }

    # Model 4: Bayesian Raw
    coefs4 <- models$model4$coefficients |> filter(str_detect(variable, "_spend"))
    for (spend_col in spend_cols) {
      channel <- str_remove(spend_col, "_spend")
      coef_row <- coefs4 |> filter(str_detect(variable, channel))
      if (nrow(coef_row) > 0) {
        coef_val <- coef_row$mean[1]
        df[[paste0(spend_col, "_contrib_model4")]] <- df[[spend_col]] * coef_val
      } else {
        df[[paste0(spend_col, "_contrib_model4")]] <- 0
      }
    }

    df
  })

  # Observer: Retrain models when button is clicked
  observeEvent(input$retrain_models, {
    output$retrain_status <- renderText("Training models... This may take a minute.")

    # Build command with parameters
    cmd <- sprintf(
      "Rscript model.R --decay_rate=%s --alpha=%s --gamma=%s",
      input$decay_rate,
      input$alpha,
      input$gamma
    )

    # Run model.R script
    result <- system(cmd, intern = TRUE)

    # Increment models_version to trigger reload
    models_version(models_version() + 1)

    output$retrain_status <- renderText("Models retrained successfully!")
  })

  # Output: Model information
  output$model_info <- renderUI({
    model_choice <- input$selected_model

    descriptions <- list(
      model1 = "Frequentist linear regression with raw marketing spend features. Provides baseline performance.",
      model2 = "Frequentist linear regression with adstock and saturation transformations. Models carryover effects and diminishing returns.",
      model3 = "Bayesian regression with transformed features. Incorporates prior knowledge and MMM transformations.",
      model4 = "Bayesian regression on raw features. Incorporates prior knowledge without transformations."
    )

    model_names <- c(
      model1 = "Simple Linear Regression",
      model2 = "Transformed Linear Regression",
      model3 = "Bayesian (Transformed)",
      model4 = "Bayesian (Raw)"
    )

    markdown(paste0(
      "### ", model_names[[model_choice]], "\n\n",
      descriptions[[model_choice]]
    ))
  })

  # Output: Model performance metrics
  output$model_performance <- renderUI({
    models <- current_models()
    model_choice <- input$selected_model

    if (model_choice == "model1") {
      metrics <- list(
        train_r2 = models$model1$train_metrics |> filter(.metric == "rsq") |> pull(.estimate),
        test_r2 = models$model1$test_metrics |> filter(.metric == "rsq") |> pull(.estimate),
        train_rmse = models$model1$train_metrics |> filter(.metric == "rmse") |> pull(.estimate),
        test_rmse = models$model1$test_metrics |> filter(.metric == "rmse") |> pull(.estimate)
      )
    } else if (model_choice == "model2") {
      metrics <- list(
        train_r2 = models$model2$train_metrics |> filter(.metric == "rsq") |> pull(.estimate),
        test_r2 = models$model2$test_metrics |> filter(.metric == "rsq") |> pull(.estimate),
        train_rmse = models$model2$train_metrics |> filter(.metric == "rmse") |> pull(.estimate),
        test_rmse = models$model2$test_metrics |> filter(.metric == "rmse") |> pull(.estimate)
      )
    } else if (model_choice == "model3") {
      metrics <- list(
        train_r2 = models$model3$train_rsq,
        test_r2 = models$model3$test_rsq,
        train_rmse = models$model3$train_rmse,
        test_rmse = models$model3$test_rmse
      )
    } else {
      metrics <- list(
        train_r2 = models$model4$train_rsq,
        test_r2 = models$model4$test_rsq,
        train_rmse = models$model4$train_rmse,
        test_rmse = models$model4$test_rmse
      )
    }

    HTML(sprintf(
      "<div class='row'>
        <div class='col-md-6'>
          <h5>Training Set</h5>
          <ul>
            <li><strong>R²:</strong> %.4f</li>
            <li><strong>RMSE:</strong> $%.2fK</li>
          </ul>
        </div>
        <div class='col-md-6'>
          <h5>Test Set</h5>
          <ul>
            <li><strong>R²:</strong> %.4f</li>
            <li><strong>RMSE:</strong> $%.2fK</li>
          </ul>
      </div>",
      metrics$train_r2, metrics$train_rmse,
      metrics$test_r2, metrics$test_rmse
    ))
  })

  # Output: KPI cards
  output$kpi_cards <- renderUI({
    df <- filtered_data()
    model_choice <- input$selected_model

    # Calculate metrics
    avg_sales <- mean(df$sales_revenue, na.rm = TRUE)
    total_spend <- df |>
      select(all_of(spend_cols)) |>
      rowSums() |>
      mean()

    # Calculate overall ROI from selected model using pre-computed contributions
    contrib_cols <- paste0(spend_cols, "_contrib_", model_choice)
    total_contribution <- df |>
      select(all_of(contrib_cols)) |>
      rowSums() |>
      sum(na.rm = TRUE)

    total_spend_sum <- df |>
      select(all_of(spend_cols)) |>
      rowSums() |>
      sum(na.rm = TRUE)

    overall_roi <- ifelse(total_spend_sum > 0, total_contribution / total_spend_sum, 0)

    layout_columns(
      value_box(
        title = "Average Weekly Sales",
        value = scales::dollar(avg_sales, scale = 1, suffix = "K", accuracy = 1),
        theme = "primary"
      ),
      value_box(
        title = "Average Weekly Spend",
        value = scales::dollar(total_spend, scale = 1, suffix = "K", accuracy = 1),
        theme = "info"
      ),
      value_box(
        title = "Overall Marketing ROI",
        value = sprintf("%.2fx", overall_roi),
        theme = "success"
      )
    )
  })

  # Output: Sales trend plot
  output$sales_trend_plot <- renderPlotly({
    df <- filtered_data()

    # Calculate total spend per week
    total_spend <- df |>
      select(all_of(spend_cols)) |>
      rowSums()

    # Create subplot
    fig <- subplot(
      plot_ly(df) |>
        add_trace(
          x = ~date,
          y = ~sales_revenue,
          type = "scatter",
          mode = "lines",
          name = "Sales Revenue",
          line = list(color = "#D01012", width = 2),
          fill = "tozeroy"
        ) |>
        layout(yaxis = list(title = "Revenue ($K)")),

      plot_ly(df) |>
        add_trace(
          x = ~date,
          y = total_spend,
          type = "scatter",
          mode = "lines",
          name = "Total Marketing Spend",
          line = list(color = "#0055BF", width = 2),
          fill = "tozeroy"
        ) |>
        layout(yaxis = list(title = "Spend ($K)")),

      nrows = 2,
      shareX = TRUE,
      titleY = TRUE,
      heights = c(0.6, 0.4),
      margin = 0.08
    ) |>
      layout(
        showlegend = FALSE,
        hovermode = "x unified",
        xaxis = list(title = ""),
        margin = list(t = 0, b = 0)
      )

    fig
  })

  # Output: Channel distribution plot
  output$channel_distribution_plot <- renderPlotly({
    df <- filtered_data()

    # Get selected channels
    selected_channels <- input$channels
    if (length(selected_channels) == 0) {
      return(NULL)
    }

    # Get spend columns for selected channels
    selected_spend_cols <- spend_cols[names(spend_cols) %in% selected_channels]

    # Calculate total spend by channel and percentage
    channel_totals <- df |>
      select(all_of(selected_spend_cols)) |>
      colSums()

    total_spend <- sum(channel_totals)

    spend_data <- tibble(
      Channel = names(selected_spend_cols),
      Spend = channel_totals,
      Percentage = (channel_totals / total_spend) * 100
    ) |>
      arrange(Spend)

    plot_ly(spend_data) |>
      add_trace(
        x = ~Percentage,
        y = ~Channel,
        type = "bar",
        orientation = "h",
        marker = list(color = "#0055BF"),
        text = ~sprintf("%.1f%%", Percentage),
        textposition = "outside",
        hovertemplate = ~sprintf("%s<br>%.1f%%<br>$%.1fK<extra></extra>", Channel, Percentage, Spend)
      ) |>
      layout(
        xaxis = list(title = "Percentage of Total Spend", range = c(0, max(spend_data$Percentage) * 1.15)),
        yaxis = list(title = "", categoryorder = "array", categoryarray = spend_data$Channel),
        showlegend = FALSE,
        margin = list(t = 0)
      )
  })

  # Output: ROI chart (calculated from selected model)
  output$roi_chart <- renderPlotly({
    df <- filtered_data()
    model_choice <- input$selected_model

    # Get selected channels
    selected_channels <- input$channels
    if (length(selected_channels) == 0) {
      return(NULL)
    }

    # Get spend columns for selected channels
    selected_spend_cols <- spend_cols[names(spend_cols) %in% selected_channels]

    # Calculate ROI for each channel using pre-computed contributions
    roi_data <- map_df(selected_spend_cols, function(spend_col) {
      channel_name <- names(spend_cols)[spend_cols == spend_col]
      contrib_col <- paste0(spend_col, "_contrib_", model_choice)

      total_spend <- sum(df[[spend_col]], na.rm = TRUE)
      total_contribution <- sum(df[[contrib_col]], na.rm = TRUE)

      roi <- ifelse(total_spend > 0, total_contribution / total_spend, 0)

      tibble(Channel = channel_name, ROI = roi)
    }) |>
      arrange(ROI)

    plot_ly(roi_data) |>
      add_trace(
        x = ~ROI,
        y = ~Channel,
        type = "bar",
        orientation = "h",
        marker = list(color = "#00A651"),
        text = ~sprintf("%.2fx", ROI),
        textposition = "outside"
      ) |>
      layout(
        xaxis = list(title = "ROI", range = c(0, max(roi_data$ROI) * 1.15)),
        yaxis = list(title = "", categoryorder = "array", categoryarray = roi_data$Channel),
        showlegend = FALSE,
        margin = list(t = 0)
      )
  })

  # Output: Channel statistics
  output$channel_stats <- renderUI({
    df <- filtered_data()
    model_choice <- input$selected_model

    channel <- input$analysis_channel
    spend_col <- spend_cols[names(spend_cols) == channel]
    contrib_col <- paste0(spend_col, "_contrib_", model_choice)

    # Calculate spend stats
    avg_spend <- mean(df[[spend_col]], na.rm = TRUE)
    total_spend <- sum(df[[spend_col]], na.rm = TRUE)

    # Calculate contribution stats
    avg_contribution <- mean(df[[contrib_col]], na.rm = TRUE)
    total_contribution <- sum(df[[contrib_col]], na.rm = TRUE)

    # Calculate ROI
    roi <- ifelse(total_spend > 0, total_contribution / total_spend, 0)

    markdown(sprintf("
### %s

**Spend:**
- Average: $%.2fK/week
- Total: $%.2fK

**Contribution:**
- Average: $%.2fK/week
- Total: $%.2fK

**ROI:** %.2fx
    ", channel, avg_spend, total_spend, avg_contribution, total_contribution, roi))
  })

  # Output: Channel scatter plot
  output$channel_scatter_plot <- renderPlotly({
    df <- filtered_data()
    model_choice <- input$selected_model

    channel <- input$analysis_channel
    spend_col <- spend_cols[names(spend_cols) == channel]
    contrib_col <- paste0(spend_col, "_contrib_", model_choice)

    plot_ly(df) |>
      add_trace(
        x = ~get(spend_col),
        y = ~get(contrib_col),
        type = "scatter",
        mode = "markers",
        marker = list(color = "#D01012", size = 8, opacity = 0.6),
        name = channel
      ) |>
      layout(
        xaxis = list(title = paste(channel, "Spend ($K)")),
        yaxis = list(title = paste(channel, "Contribution ($K)")),
        showlegend = FALSE,
        margin = list(t = 20)
      )
  })

  # Output: Channel time series
  output$channel_time_series <- renderPlotly({
    df <- filtered_data()
    model_choice <- input$selected_model

    channel <- input$analysis_channel
    spend_col <- spend_cols[names(spend_cols) == channel]
    contrib_col <- paste0(spend_col, "_contrib_", model_choice)

    # Create dual-axis time series
    fig <- plot_ly(df)

    fig <- fig |>
      add_trace(
        x = ~date,
        y = ~get(spend_col),
        type = "scatter",
        mode = "lines",
        name = "Spend",
        line = list(color = "#0055BF", width = 2),
        yaxis = "y"
      ) |>
      add_trace(
        x = ~date,
        y = ~get(contrib_col),
        type = "scatter",
        mode = "lines",
        name = "Contribution",
        line = list(color = "#00A651", width = 2),
        yaxis = "y2"
      ) |>
      layout(
        xaxis = list(title = ""),
        yaxis = list(
          title = "Spend ($K)",
          side = "left"
        ),
        yaxis2 = list(
          title = "Contribution ($K)",
          overlaying = "y",
          side = "right"
        ),
        hovermode = "x unified",
        legend = list(x = 0.1, y = 1),
        margin = list(t = 20)
      )

    fig
  })

  # Reactive: Optimized budget allocation
  optimized_budget <- eventReactive(input$optimize, {
    models <- current_models()
    model_choice <- input$selected_model
    total_budget <- input$total_budget
    min_pct <- input$min_allocation / 100
    max_pct <- input$max_allocation / 100

    # Get coefficients from selected model
    if (model_choice == "model1") {
      coefs <- models$model1$coefficients |> filter(str_detect(term, "_spend"))
    } else if (model_choice == "model2") {
      coefs <- models$model2$coefficients |> filter(str_detect(term, "_transformed"))
    } else if (model_choice == "model3") {
      coefs <- models$model3$coefficients |> filter(str_detect(variable, "_transformed"))
    } else {
      coefs <- models$model4$coefficients |> filter(str_detect(variable, "_spend"))
    }

    # Calculate importance for each channel
    importance_data <- map_df(spend_cols, function(spend_col) {
      channel_base <- str_remove(spend_col, "_spend")

      if (model_choice %in% c("model1", "model2")) {
        coef_row <- coefs |> filter(str_detect(term, channel_base))
        if (nrow(coef_row) > 0) {
          coef_val <- coef_row$estimate[1]
        } else {
          coef_val <- NA
        }
      } else {
        coef_row <- coefs |> filter(str_detect(variable, channel_base))
        if (nrow(coef_row) > 0) {
          coef_val <- coef_row$mean[1]
        } else {
          coef_val <- NA
        }
      }

      tibble(
        Channel = names(spend_cols)[spend_cols == spend_col],
        coefficient = ifelse(!is.na(coef_val) && coef_val > 0, coef_val, 0)
      )
    })

    # Filter positive coefficients only
    importance_data <- importance_data |> filter(coefficient > 0)

    if (nrow(importance_data) == 0) {
      return(tibble(
        Channel = character(),
        `Budget %` = numeric(),
        `Recommended Spend ($K)` = numeric()
      ))
    }

    # Normalize coefficients to get allocation weights
    total_coef <- sum(importance_data$coefficient)
    importance_data <- importance_data |>
      mutate(weight = coefficient / total_coef)

    # Apply constraints (min/max allocation)
    importance_data <- importance_data |>
      mutate(weight = pmax(pmin(weight, max_pct), min_pct))

    # Renormalize after clipping
    importance_data <- importance_data |>
      mutate(weight = weight / sum(weight))

    # Calculate recommended spend
    importance_data <- importance_data |>
      mutate(
        `Budget %` = weight * 100,
        `Recommended Spend ($K)` = weight * total_budget
      ) |>
      select(Channel, `Budget %`, `Recommended Spend ($K)`) |>
      arrange(desc(`Recommended Spend ($K)`))

    importance_data
  })

  # Output: Optimization results table
  output$optimization_results <- renderUI({
    req(input$optimize)

    result <- optimized_budget()

    if (nrow(result) == 0) {
      return(p("No positive channel coefficients found. Try a different model."))
    }

    # Create styled table rows
    table_rows <- result |>
      mutate(
        row = sprintf(
          "<tr><td>%s</td><td>%.1f%%</td><td>$%.2fK</td></tr>",
          Channel, `Budget %`, `Recommended Spend ($K)`
        )
      ) |>
      pull(row) |>
      paste(collapse = "")

    html_table <- sprintf(
      "<div class='table-responsive'>
        <table class='table table-striped'>
          <thead>
            <tr>
              <th>Channel</th>
              <th>Budget %%</th>
              <th>Recommended Spend</th>
            </tr>
          </thead>
          <tbody>
            %s
          </tbody>
        </table>
      </div>
      <p class='mt-3'><strong>Total Budget: $%.2fK per week</strong></p>",
      table_rows,
      input$total_budget
    )

    HTML(html_table)
  })

  # Output: Allocation comparison chart
  output$allocation_comparison <- renderPlotly({
    req(input$optimize)

    result <- optimized_budget()
    df <- filtered_data()

    if (nrow(result) == 0) {
      return(NULL)
    }

    # Calculate current average allocation
    current_alloc <- map_df(result$Channel, function(channel) {
      spend_col <- spend_cols[names(spend_cols) == channel]
      tibble(
        Channel = channel,
        Current = mean(df[[spend_col]], na.rm = TRUE)
      )
    })

    # Combine with recommended
    comparison <- result |>
      left_join(current_alloc, by = "Channel")

    plot_ly(comparison) |>
      add_trace(
        x = ~Channel,
        y = ~Current,
        type = "bar",
        name = "Current Avg",
        marker = list(color = "#6B6B6B")
      ) |>
      add_trace(
        x = ~Channel,
        y = ~`Recommended Spend ($K)`,
        type = "bar",
        name = "Recommended",
        marker = list(color = "#00A651")
      ) |>
      layout(
        barmode = "group",
        xaxis = list(title = ""),
        yaxis = list(title = "Weekly Spend ($K)"),
        legend = list(x = 0.1, y = 1),
        margin = list(t = 20)
      )
  })
}

# Run app
shinyApp(ui, server)
