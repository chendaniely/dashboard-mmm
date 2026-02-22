# LLM Usage note:
# linear regression modeling was done by hand,
# but the bayesian model, transformations (adstock and saturation),
# print/cat staements, and code comments
# were put in by Claude Code.
#
# This script is meade to be run:
# 1. standalone in an interactive session
# 2. command line with arguments
# 3. sourced within a shiny app to change the parameters
# This is not the best way to make model modification in a shiny app
# it was made for a workflow demo that separates the modeling from other
# data science artifacts.

# ============================================================================
# PROGRESSIVE MMM MODELING TUTORIAL
# ============================================================================
# This file demonstrates four progressive approaches to Marketing Mix Modeling:
#
# 1. Simple Linear Regression (Baseline)
#    - Frequentist linear regression with raw marketing spend features
#    - Provides baseline performance to compare against
#
# 2. Linear Regression with Transformed Features
#    - Frequentist linear regression with MMM-specific transformations:
#      * Adstock: Models carryover effects (marketing impact over time)
#      * Saturation: Models diminishing returns (impact decreases with spend)
#    - Shows the impact of proper feature engineering on model performance
#
# 3. Bayesian Linear Regression (Raw Features)
#    - Bayesian regression on raw marketing spend using rstanarm
#    - Incorporates prior knowledge about expected coefficient ranges
#    - Shows the impact of Bayesian approach without transformations
#
# 4. Bayesian Linear Regression (Transformed Features)
#    - Combines both Bayesian priors AND MMM transformations
#    - Represents the most sophisticated approach
#    - Best practice for production MMM models
#
# All models control for seasonality using the holiday_week indicator variable

library(tidyverse)
library(tidymodels)
library(rstanarm)

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================
# Parse command-line arguments if provided, otherwise use defaults
# This allows the script to be run standalone OR called from Shiny/terminal
#
# Command-line usage examples:
#   Rscript model.R
#   Rscript model.R --decay_rate=0.5 --alpha=1.2 --gamma=0.8
#
# Shiny integration examples:
#   Adstock decay rate widget:
#     sliderInput("decay_rate", "Decay Rate", min = 0, max = 1, value = 0.4, step = 0.1)
#   Saturation alpha widget:
#     numericInput("alpha", "Alpha (Scale)", value = 1, min = 0.1, max = 5, step = 0.1)
#   Saturation gamma widget:
#     sliderInput("gamma", "Gamma (Shape)", min = 0.1, max = 1.5, value = 0.7, step = 0.1)
#   Call script from Shiny:
#     system(sprintf("Rscript model.R --decay_rate=%s --alpha=%s --gamma=%s",
#                    input$decay_rate, input$alpha, input$gamma))

args <- commandArgs(trailingOnly = TRUE)

# Default transformation parameter values
decay_rate <- 0.4  # Adstock decay rate: How much effect carries over week-to-week (0-1)
alpha <- 1         # Saturation alpha: Maximum possible effect (scaling factor)
gamma <- 0.7       # Saturation gamma: Shape of diminishing returns curve (0.3-1.5 typical)

# Parse command-line arguments and override defaults
if (length(args) > 0) {
  for (arg in args) {
    if (grepl("^--decay_rate=", arg)) {
      decay_rate <- as.numeric(sub("^--decay_rate=", "", arg))
    } else if (grepl("^--alpha=", arg)) {
      alpha <- as.numeric(sub("^--alpha=", "", arg))
    } else if (grepl("^--gamma=", arg)) {
      gamma <- as.numeric(sub("^--gamma=", "", arg))
    }
  }
  cat("\n=== Using Command-Line Parameters ===\n")
  cat("  decay_rate:", decay_rate, "(Adstock carryover effect)\n")
  cat("  alpha:", alpha, "(Saturation scale)\n")
  cat("  gamma:", gamma, "(Diminishing returns shape)\n\n")
} else {
  cat("\n=== Using Default Parameters ===\n")
  cat("  decay_rate:", decay_rate, "(Adstock carryover effect)\n")
  cat("  alpha:", alpha, "(Saturation scale)\n")
  cat("  gamma:", gamma, "(Diminishing returns shape)\n")
  cat("  (Run with --decay_rate=X --alpha=Y --gamma=Z to override)\n\n")
}

# ============================================================================
# LOAD DATA
# ============================================================================

data <- read_csv("data/synthetic-marketing-sales.csv")
data

data_model <- data |>
  select(date, week, year, quarter, month, week_of_year, holiday_week, sales_revenue,
    ends_with("_spend"),
    -total_marketing_spend
  ) |>
  mutate(holiday_week = factor(holiday_week, levels = c(0, 1), labels = c("No", "Yes")))
data_model

# Train/test split (80/20) using tidymodels
# IMPORTANT: For time series data, we need temporal split (not random)
# Training = first 80% chronologically, Testing = last 20%
# This prevents data leakage (don't train on future, test on past)
data_split <- initial_time_split(data_model, prop = 0.8)
train_data <- training(data_split)
test_data <- testing(data_split)

cat("\n=== Data Split ===\n")
cat("Training samples:", nrow(train_data), "\n")
cat("Testing samples:", nrow(test_data), "\n")

# ============================================================================
# APPROACH 1: Simple Linear Regression
# ============================================================================
cat("\n\n=== APPROACH 1: Simple Linear Regression ===\n")

lr_spec <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

lr_recipe <- recipe(sales_revenue ~ ., data = train_data) |>
  update_role(date, week, year, quarter, month, week_of_year, new_role = "id") |>
  step_dummy(holiday_week, one_hot = FALSE) |>
  step_normalize(all_numeric_predictors())

lr_workflow <- workflow() |>
  add_model(lr_spec) |>
  add_recipe(lr_recipe)

lr_fit <- lr_workflow |>
  fit(data = train_data)

lr_train_pred <- predict(lr_fit, train_data) |>
  bind_cols(train_data |> select(sales_revenue))

lr_test_pred <- predict(lr_fit, test_data) |>
  bind_cols(test_data |> select(sales_revenue))

lr_train_metrics <- lr_train_pred |>
  metrics(truth = sales_revenue, estimate = .pred)

lr_test_metrics <- lr_test_pred |>
  metrics(truth = sales_revenue, estimate = .pred)

cat("\nSimple LR Training Metrics:\n")
print(lr_train_metrics)
cat("\nSimple LR Testing Metrics:\n")
print(lr_test_metrics)

lr_coefs <- lr_fit |>
  extract_fit_engine() |>
  tidy() |>
  arrange(desc(abs(estimate)))

cat("\nLinear Regression Coefficients:\n")
print(lr_coefs |> head(100))

# ============================================================================
# APPROACH 2: Linear Regression with Transformed Features
# ============================================================================
cat("\n\n=== APPROACH 2: Linear Regression with Feature Transformations ===\n")

# ----------------------------------------------------------------------------
# PARAMETER TUNING NOTE
# ----------------------------------------------------------------------------
# The parameters below (decay_rate=0.4, gamma=0.7) are starting values
# In practice, these should be tuned through:
# 1. Grid search or Bayesian optimization
# 2. Cross-validation to prevent overfitting
# 3. Business knowledge (e.g., brand campaigns need longer decay)
#
# Common parameter ranges:
# - decay_rate: 0.2-0.8 (higher for brand, lower for performance marketing)
# - gamma: 0.3-1.0 (lower = stronger diminishing returns)

# ============================================================================
# ADSTOCK TRANSFORMATION (Carryover Effect)
# ============================================================================
# Purpose: Models the delayed impact of marketing spend over time
# Real-world example: A TV ad aired this week continues to influence purchases
#                     for several weeks as people remember the message
#
# Formula: adstock[t] = spend[t] + decay_rate * adstock[t-1]
#
# How it works:
# - Each week's adstock includes current spend + a decayed portion of past adstock
# - decay_rate = 0.4 means 40% of last week's effect carries forward
# - This creates exponential decay: week 1 = 100%, week 2 = 40%, week 3 = 16%, etc.
#
# Parameters:
# - decay_rate: How much of the effect persists (0-1 range)
#   * Higher = longer carryover (e.g., 0.7 for brand awareness campaigns)
#   * Lower = shorter carryover (e.g., 0.3 for tactical promotions)
#
# Why we need this:
# - Marketing attribution problem: Sales today might be from ads last week/month
# - Without adstock, model undervalues marketing that has delayed effects
# - Captures the "memory" effect of advertising
apply_adstock <- function(x, decay_rate = 0.5) {
  n <- length(x)
  result <- numeric(n)
  result[1] <- x[1]  # First week has no carryover from past
  for (i in 2:n) {
    # Current week = this week's spend + decay from previous accumulated effect
    result[i] <- x[i] + decay_rate * result[i-1]
  }
  return(result)
}

# ============================================================================
# SATURATION TRANSFORMATION (Diminishing Returns)
# ============================================================================
# Purpose: Models how marketing effectiveness decreases as spend increases
# Real-world example: First $10K in TV ads reaches new people, but the 10th $10K
#                     reaches mostly people who already saw the ad (lower ROI)
#
# Formula: Hill function = alpha * x^gamma / (1 + x^gamma)
#
# How it works:
# - S-shaped curve that starts slow, accelerates, then plateaus
# - At low spend: linear growth (high marginal ROI)
# - At high spend: flattens out (low marginal ROI, "saturated")
#
# Parameters:
# - alpha: Maximum possible effect (scaling factor)
# - gamma: Shape of the curve (typically 0.3 to 1.0)
#   * gamma < 1: Strong diminishing returns (curve flattens quickly)
#   * gamma = 1: Moderate diminishing returns
#   * gamma > 1: Delayed saturation (S-curve more pronounced)
#
# Why we need this:
# - Linear models assume $1 has the same impact regardless of total spend
# - In reality, there's a saturation point (can't show same person 100 ads)
# - Captures the concept of "market saturation" and "frequency fatigue"
# - Essential for budget optimization (helps find optimal spend levels)
apply_saturation <- function(x, alpha = 1, gamma = 0.5) {
  # Normalize input to [0,1] range for numerical stability
  x_scaled <- x / max(x, na.rm = TRUE)

  # Hill function: alpha controls max, gamma controls curve shape
  # Result: Output grows with input but at a decreasing rate
  alpha * (x_scaled^gamma) / (1 + x_scaled^gamma)
}

# ============================================================================
# APPLYING TRANSFORMATIONS: Order Matters!
# ============================================================================
# We apply transformations in a specific order: ADSTOCK → SATURATION
#
# Why this order?
# 1. ADSTOCK FIRST: Accumulates spend over time (raw dollars → effective exposure)
#    - Input: Weekly marketing spend in dollars
#    - Output: Accumulated advertising exposure
#
# 2. SATURATION SECOND: Models diminishing returns on accumulated exposure
#    - Input: Accumulated exposure from adstock
#    - Output: Actual impact on sales (saturated effect)
#
# Think of it as: Raw Spend → Memory Effect → Diminishing Returns → Sales Impact
#
# Example with TV ads:
# - Week 1: Spend $50K → Adstock $50K → Saturation applies to $50K
# - Week 2: Spend $50K → Adstock $70K (50 + 0.4*50) → Saturation applies to $70K
# - The $70K accumulated exposure experiences diminishing returns, not the raw $50K
#
spend_cols <- names(train_data)[str_detect(names(train_data), "_spend$")]

# Transform training data using tidyverse
train_transformed <- train_data |>
  # STEP 1: Apply adstock to all spend columns
  # Converts raw weekly spend into accumulated advertising exposure
  mutate(across(
    ends_with("_spend"),
    ~apply_adstock(., decay_rate = decay_rate),
    .names = "{.col}_adstock"
  )) |>
  # STEP 2: Apply saturation to all adstock columns
  # Models diminishing returns on the accumulated exposure
  mutate(across(
    ends_with("_adstock"),
    ~apply_saturation(., alpha = alpha, gamma = gamma),
    .names = "{str_remove(.col, '_adstock')}_transformed"
  ))

# ============================================================================
# TRANSFORM TEST DATA: Preventing Data Leakage
# ============================================================================
# CRITICAL: Test data transformations must continue from training, not reset!
#
# Why this matters:
# - Adstock is cumulative and has "memory" from past weeks
# - If we reset adstock at the test split, we'd incorrectly assume no carryover
# - This would give unrealistic test predictions (data leakage in reverse)
#
# Example of the problem:
# - Last week of training: Big holiday campaign with high adstock
# - First week of testing: No new spend, but should still have carryover effect
# - If we reset: Model sees 0 adstock → predicts low sales (WRONG)
# - If we continue: Model sees decayed adstock → predicts correctly
#

# Extract last adstock values from training data (for continuity)
last_adstock_values <- train_transformed |>
  select(ends_with("_adstock")) |>
  slice_tail(n = 1) |>
  as.list()

# Transform test data using tidyverse
test_transformed <- test_data |>
  # STEP 1: Apply adstock to all spend columns (continuing from training)
  mutate(across(
    ends_with("_spend"),
    ~{
      # Get the corresponding last adstock value from training
      col_name <- cur_column()
      adstock_col <- paste0(col_name, "_adstock")
      last_val <- last_adstock_values[[adstock_col]]

      # Prepend last training value, apply transformation, remove first element
      test_vals <- c(last_val, .)
      test_adstock <- apply_adstock(test_vals, decay_rate = decay_rate)
      test_adstock[-1]  # Remove the prepended value
    },
    .names = "{.col}_adstock"
  )) |>
  # STEP 2: Apply saturation to all adstock columns
  # Note: Saturation is applied independently (no memory from training needed)
  mutate(across(
    ends_with("_adstock"),
    ~apply_saturation(., alpha = alpha, gamma = gamma),
    .names = "{str_remove(.col, '_adstock')}_transformed"
  ))

# Select transformed features for modeling
transformed_cols <- names(train_transformed)[str_detect(names(train_transformed), "_transformed$")]

lr_transformed_recipe <- recipe(sales_revenue ~ ., data = train_transformed) |>
  update_role(date, week, year, quarter, month, week_of_year, new_role = "id") |>
  step_rm(all_of(spend_cols)) |>  # Don't use raw spend
  step_rm(ends_with("_adstock")) |>  # Don't use intermediate adstock
  step_dummy(holiday_week, one_hot = FALSE) |>
  step_normalize(all_numeric_predictors())

lr_transformed_workflow <- workflow() |>
  add_model(lr_spec) |>
  add_recipe(lr_transformed_recipe)

lr_transformed_fit <- lr_transformed_workflow |>
  fit(data = train_transformed)

lr_trans_train_pred <- predict(lr_transformed_fit, train_transformed) |>
  bind_cols(train_transformed |> select(sales_revenue))

lr_trans_test_pred <- predict(lr_transformed_fit, test_transformed) |>
  bind_cols(test_transformed |> select(sales_revenue))

# Evaluate
lr_trans_train_metrics <- lr_trans_train_pred |>
  metrics(truth = sales_revenue, estimate = .pred)

lr_trans_test_metrics <- lr_trans_test_pred |>
  metrics(truth = sales_revenue, estimate = .pred)

cat("\nTransformed LR Training Metrics:\n")
print(lr_trans_train_metrics)
cat("\nTransformed LR Testing Metrics:\n")
print(lr_trans_test_metrics)

lr_trans_coefs <- lr_transformed_fit |>
  extract_fit_engine() |>
  tidy() |>
  arrange(desc(abs(estimate)))

cat("\nTop Transformed Feature Coefficients:\n")
print(lr_trans_coefs |> head(100))

# ============================================================================
# APPROACH 3: Bayesian Linear Regression
# ============================================================================
cat("\n\n=== APPROACH 3: Bayesian Linear Regression ===\n")

# Prepare data for rstanarm (use transformed features + holiday_week)
# holiday_week is important to control for seasonality effects
train_bayes <- train_transformed |>
  select(sales_revenue, all_of(transformed_cols), holiday_week)

test_bayes <- test_transformed |>
  select(sales_revenue, all_of(transformed_cols), holiday_week)

# Standardize predictors manually for better priors
# NOTE: We only scale the transformed marketing features, NOT holiday_week
# holiday_week stays as 0/1 binary indicator
train_bayes_scaled <- train_bayes |>
  mutate(across(all_of(transformed_cols), ~scale(.)[,1]))

test_bayes_scaled <- test_bayes |>
  mutate(across(all_of(transformed_cols), ~scale(.)[,1]))

# Train Bayesian model with informative priors
# Prior: coefficients should be positive (marketing increases sales)
# Using normal prior centered at 50 with sd of 30 (weakly informative)
bayes_fit <- stan_glm(
  sales_revenue ~ .,
  data = train_bayes_scaled,
  family = gaussian(),
  prior = normal(location = 50, scale = 30),  # Weakly informative prior
  prior_intercept = normal(location = 500, scale = 100),  # Prior for baseline sales
  chains = 2,
  iter = 1000,
  seed = 42,
  refresh = 0  # Suppress iteration messages
)

cat("\nBayesian Model Summary:\n")
print(summary(bayes_fit))

# Make predictions
bayes_train_pred <- predict(bayes_fit, newdata = train_bayes_scaled)
bayes_test_pred <- predict(bayes_fit, newdata = test_bayes_scaled)

# Calculate metrics manually
bayes_train_rmse <- sqrt(mean((train_bayes$sales_revenue - bayes_train_pred)^2))
bayes_train_rsq <- cor(train_bayes$sales_revenue, bayes_train_pred)^2

bayes_test_rmse <- sqrt(mean((test_bayes$sales_revenue - bayes_test_pred)^2))
bayes_test_rsq <- cor(test_bayes$sales_revenue, bayes_test_pred)^2

cat("\nBayesian Model Training Metrics:\n")
cat("RMSE:", round(bayes_train_rmse, 2), "\n")
cat("R²:", round(bayes_train_rsq, 4), "\n")

cat("\nBayesian Model Testing Metrics:\n")
cat("RMSE:", round(bayes_test_rmse, 2), "\n")
cat("R²:", round(bayes_test_rsq, 4), "\n")

# Posterior distributions of coefficients
bayes_coefs <- as.data.frame(bayes_fit) |>
  summarise(across(everything(), list(mean = mean, sd = sd))) |>
  pivot_longer(everything(), names_to = "param", values_to = "value") |>
  separate(param, into = c("variable", "stat"), sep = "_(?=mean|sd)") |>
  pivot_wider(names_from = stat, values_from = value) |>
  arrange(desc(abs(mean)))

cat("\nBayesian Coefficient Estimates (Posterior Means):\n")
print(bayes_coefs |> head(100))

# ============================================================================
# APPROACH 4: Bayesian Linear Regression (Non-Transformed Features)
# ============================================================================
cat("\n\n=== APPROACH 4: Bayesian Linear Regression on Non-Transformed Data ===\n")

# Prepare data for rstanarm (use raw spend features + holiday_week)
# holiday_week is important to control for seasonality effects
train_bayes_raw <- train_data |>
  select(sales_revenue, all_of(spend_cols), holiday_week)

test_bayes_raw <- test_data |>
  select(sales_revenue, all_of(spend_cols), holiday_week)

# Standardize predictors manually for better priors
# NOTE: We only scale the raw spend features, NOT holiday_week
# holiday_week stays as 0/1 binary indicator
train_bayes_raw_scaled <- train_bayes_raw |>
  mutate(across(all_of(spend_cols), ~scale(.)[,1]))

test_bayes_raw_scaled <- test_bayes_raw |>
  mutate(across(all_of(spend_cols), ~scale(.)[,1]))

# Train Bayesian model with informative priors on raw features
cat("Training Bayesian regression on raw features (this may take a minute)...\n")
bayes_raw_fit <- stan_glm(
  sales_revenue ~ .,
  data = train_bayes_raw_scaled,
  family = gaussian(),
  prior = normal(location = 50, scale = 30),  # Weakly informative prior
  prior_intercept = normal(location = 500, scale = 100),  # Prior for baseline sales
  chains = 2,
  iter = 1000,
  seed = 42,
  refresh = 0  # Suppress iteration messages
)

cat("\nBayesian Model (Raw Features) Summary:\n")
print(summary(bayes_raw_fit))

# Make predictions
bayes_raw_train_pred <- predict(bayes_raw_fit, newdata = train_bayes_raw_scaled)
bayes_raw_test_pred <- predict(bayes_raw_fit, newdata = test_bayes_raw_scaled)

# Calculate metrics manually
bayes_raw_train_rmse <- sqrt(mean((train_bayes_raw$sales_revenue - bayes_raw_train_pred)^2))
bayes_raw_train_rsq <- cor(train_bayes_raw$sales_revenue, bayes_raw_train_pred)^2

bayes_raw_test_rmse <- sqrt(mean((test_bayes_raw$sales_revenue - bayes_raw_test_pred)^2))
bayes_raw_test_rsq <- cor(test_bayes_raw$sales_revenue, bayes_raw_test_pred)^2

cat("\nBayesian Model (Raw) Training Metrics:\n")
cat("RMSE:", round(bayes_raw_train_rmse, 2), "\n")
cat("R²:", round(bayes_raw_train_rsq, 4), "\n")

cat("\nBayesian Model (Raw) Testing Metrics:\n")
cat("RMSE:", round(bayes_raw_test_rmse, 2), "\n")
cat("R²:", round(bayes_raw_test_rsq, 4), "\n")

# Posterior distributions of coefficients
bayes_raw_coefs <- as.data.frame(bayes_raw_fit) |>
  summarise(across(everything(), list(mean = mean, sd = sd))) |>
  pivot_longer(everything(), names_to = "param", values_to = "value") |>
  separate(param, into = c("variable", "stat"), sep = "_(?=mean|sd)") |>
  pivot_wider(names_from = stat, values_from = value) |>
  arrange(desc(abs(mean)))

cat("\nBayesian (Raw) Coefficient Estimates (Posterior Means):\n")
print(bayes_raw_coefs |> head(10))

# ============================================================================
# MODEL COMPARISON
# ============================================================================
cat("\n\n=== MODEL COMPARISON ===\n")

comparison <- tibble(
  Model = c("1. Simple LR", "2. Transformed LR", "3. Bayesian LR (Raw)", "4. Bayesian LR (Transformed)"),
  Train_RMSE = c(
    lr_train_metrics |> filter(.metric == "rmse") |> pull(.estimate),
    lr_trans_train_metrics |> filter(.metric == "rmse") |> pull(.estimate),
    bayes_raw_train_rmse,
    bayes_train_rmse
  ),
  Test_RMSE = c(
    lr_test_metrics |> filter(.metric == "rmse") |> pull(.estimate),
    lr_trans_test_metrics |> filter(.metric == "rmse") |> pull(.estimate),
    bayes_raw_test_rmse,
    bayes_test_rmse
  ),
  Train_R2 = c(
    lr_train_metrics |> filter(.metric == "rsq") |> pull(.estimate),
    lr_trans_train_metrics |> filter(.metric == "rsq") |> pull(.estimate),
    bayes_raw_train_rsq,
    bayes_train_rsq
  ),
  Test_R2 = c(
    lr_test_metrics |> filter(.metric == "rsq") |> pull(.estimate),
    lr_trans_test_metrics |> filter(.metric == "rsq") |> pull(.estimate),
    bayes_raw_test_rsq,
    bayes_test_rsq
  )
)

print(comparison)

cat("\n=== Key Insights ===\n")
cat("1. Simple LR: Baseline model with raw features\n")
cat("2. Transformed LR: Shows impact of adstock + saturation transformations\n")
cat("3. Bayesian LR (Raw): Shows impact of Bayesian priors without transformations\n")
cat("4. Bayesian LR (Transformed): Combines both transformations and priors\n")
cat("\nLower RMSE and Higher R² indicate better model performance.\n")

cat("\n=== Analysis Complete ===\n")

# ============================================================================
# SAVE MODELS FOR SHINY APPLICATION
# ============================================================================
cat("\n\n=== Saving Models ===\n")

# Create output directory
output_dir <- "saved_models"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Save transformation parameters
transformation_params <- list(
  decay_rate = decay_rate,
  alpha = alpha,
  gamma = gamma
)
saveRDS(transformation_params, file.path(output_dir, "transformation_params.rds"))
cat("✓ Saved transformation parameters\n")

# Save Model 1: Simple Linear Regression
model1_objects <- list(
  workflow = lr_workflow,
  fit = lr_fit,
  recipe = lr_recipe,
  train_metrics = lr_train_metrics,
  test_metrics = lr_test_metrics,
  coefficients = lr_coefs
)
saveRDS(model1_objects, file.path(output_dir, "model1_simple_lr.rds"))
cat("✓ Saved Model 1: Simple Linear Regression\n")

# Save Model 2: Transformed Linear Regression
model2_objects <- list(
  workflow = lr_transformed_workflow,
  fit = lr_transformed_fit,
  recipe = lr_transformed_recipe,
  train_data = train_transformed,
  test_data = test_transformed,
  train_metrics = lr_trans_train_metrics,
  test_metrics = lr_trans_test_metrics,
  coefficients = lr_trans_coefs,
  transformed_cols = transformed_cols
)
saveRDS(model2_objects, file.path(output_dir, "model2_transformed_lr.rds"))
cat("✓ Saved Model 2: Transformed Linear Regression\n")

# Save Model 3: Bayesian LR (Transformed)
model3_objects <- list(
  fit = bayes_fit,
  train_data = train_bayes_scaled,
  test_data = test_bayes_scaled,
  train_rmse = bayes_train_rmse,
  test_rmse = bayes_test_rmse,
  train_rsq = bayes_train_rsq,
  test_rsq = bayes_test_rsq,
  coefficients = bayes_coefs,
  transformed_cols = transformed_cols
)
saveRDS(model3_objects, file.path(output_dir, "model3_bayes_transformed.rds"))
cat("✓ Saved Model 3: Bayesian LR (Transformed)\n")

# Save Model 4: Bayesian LR (Raw)
model4_objects <- list(
  fit = bayes_raw_fit,
  train_data = train_bayes_raw_scaled,
  test_data = test_bayes_raw_scaled,
  train_rmse = bayes_raw_train_rmse,
  test_rmse = bayes_raw_test_rmse,
  train_rsq = bayes_raw_train_rsq,
  test_rsq = bayes_raw_test_rsq,
  coefficients = bayes_raw_coefs,
  spend_cols = spend_cols
)
saveRDS(model4_objects, file.path(output_dir, "model4_bayes_raw.rds"))
cat("✓ Saved Model 4: Bayesian LR (Raw)\n")

# Save model comparison results
comparison_results <- list(
  comparison_table = comparison,
  train_data_split = train_data,
  test_data_split = test_data
)
saveRDS(comparison_results, file.path(output_dir, "comparison_results.rds"))
cat("✓ Saved model comparison results\n")

# Save a metadata file with information about the models
metadata <- list(
  date_created = Sys.time(),
  transformation_params = transformation_params,
  data_info = list(
    train_samples = nrow(train_data),
    test_samples = nrow(test_data),
    train_date_range = c(min(train_data$date), max(train_data$date)),
    test_date_range = c(min(test_data$date), max(test_data$date))
  ),
  feature_info = list(
    spend_cols = spend_cols,
    transformed_cols = transformed_cols
  ),
  model_performance = comparison
)
saveRDS(metadata, file.path(output_dir, "metadata.rds"))
cat("✓ Saved metadata\n")

cat("\nAll models saved to:", output_dir, "\n")
cat("\nTo load in Shiny app:\n")
cat("  model1 <- readRDS('saved_models/model1_simple_lr.rds')\n")
cat("  model2 <- readRDS('saved_models/model2_transformed_lr.rds')\n")
cat("  model3 <- readRDS('saved_models/model3_bayes_transformed.rds')\n")
cat("  model4 <- readRDS('saved_models/model4_bayes_raw.rds')\n")
cat("  params <- readRDS('saved_models/transformation_params.rds')\n")
cat("  metadata <- readRDS('saved_models/metadata.rds')\n")
