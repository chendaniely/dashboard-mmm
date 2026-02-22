# Marketing Mix Modeling (MMM)

A progressive approach to Marketing Mix Modeling using synthetic marketing and sales data for a hypothetical company, BrickCraft Studios.

## Overview

This project demonstrates four different approaches to Marketing Mix Modeling, from simple linear regression to sophisticated Bayesian models with marketing-specific transformations (adstock and saturation effects).

## Models

1. **Simple Linear Regression** - Baseline model with raw marketing spend features
2. **Transformed Linear Regression** - Incorporates adstock (carryover effects) and saturation (diminishing returns) transformations
3. **Bayesian Linear Regression (Raw)** - Bayesian approach with raw features and informative priors
4. **Bayesian Linear Regression (Transformed)** - Combines Bayesian inference with MMM transformations (best practice)

## Features

- Progressive modeling approach showing impact of feature engineering and Bayesian methods
- Marketing-specific transformations:
  - **Adstock**: Models delayed/carryover effects of marketing spend
  - **Saturation**: Models diminishing returns at higher spend levels
- Command-line configurable transformation parameters
- Designed for integration with Shiny dashboards
- Comprehensive EDA and model analysis using Quarto

## Project Structure

```
├── data/                           # Marketing and sales data
├── saved_models/                   # Trained model artifacts
├── data-pull.R                    # Data pipeline (demo)
├── model.R                        # Main modeling script (4 approaches)
├── eda.qmd                        # Exploratory data analysis
├── model-analysis.qmd             # Model diagnostics and comparison
└── README.md
```

### Key Files

- **`data-pull.R`**: Demonstration data pipeline showing how to pull and combine data from multiple sources (databases, marketing APIs, Google Sheets, CSV files). For demonstration purposes only - shows typical MMM data engineering workflow.

- **`model.R`**: Main modeling script that trains 4 progressive MMM approaches. Can be run standalone, from command line with parameters, or sourced from Shiny apps.

- **`eda.qmd`**: Exploratory data analysis report examining marketing channels, sales patterns, and data quality.

- **`model-analysis.qmd`**: Post-modeling analysis with diagnostics, actual vs predicted plots, and recommendations for next steps.

## Setup

This project uses `renv` for package management. To set up:

```r
# Install renv if needed
install.packages("renv")

# Restore project dependencies
renv::restore()
```

## Usage

### Run with default parameters

```bash
Rscript model.R
```

### Run with custom transformation parameters

```bash
Rscript model.R --decay_rate=0.5 --alpha=1.2 --gamma=0.8
```

**Parameters:**
- `decay_rate` (0-1): Adstock carryover effect (higher = longer memory)
- `alpha`: Saturation scale parameter (maximum effect)
- `gamma` (0.3-1.5): Saturation shape (lower = stronger diminishing returns)

### View EDA and Analysis

Render the analysis documents:

```bash
quarto render eda.qmd
quarto render model-analysis.qmd
```

## Model Output

Models are automatically saved to `saved_models/` directory:
- Individual model objects (`.rds` files)
- Transformation parameters
- Comparison metrics
- Metadata with performance statistics

## Integration

Models are designed to be loaded into Shiny applications:

```r
model1 <- readRDS('saved_models/model1_simple_lr.rds')
model2 <- readRDS('saved_models/model2_transformed_lr.rds')
params <- readRDS('saved_models/transformation_params.rds')
metadata <- readRDS('saved_models/metadata.rds')
```

## Requirements

- R >= 4.5
- tidyverse
- tidymodels
- rstanarm
- Additional packages managed via `renv`

## Workflow

1. **Data Pull** (`data-pull.R`) - Combine data from multiple sources
2. **Exploratory Analysis** (`eda.qmd`) - Understand patterns and relationships
3. **Modeling** (`model.R`) - Train and compare 4 MMM approaches
4. **Model Analysis** (`model-analysis.qmd`) - Diagnostics and evaluation
5. **Deployment** - Load models into Shiny dashboard

## Credits

Linear regression modeling was done manually. Bayesian models, transformations (adstock and saturation), data pipeline script, and code documentation were created with assistance from Claude Code.
