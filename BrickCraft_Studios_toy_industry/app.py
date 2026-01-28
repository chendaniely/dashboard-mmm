"""
Marketing Mix Modeling Dashboard
BrickCraft Studios

Interactive dashboard for exploring MMM results, channel performance, and budget optimization.

This project contains synthetic data and analysis created for demonstration purposes only.
"""

from shiny import App, reactive, render, ui
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path

# Add ml directory to path
sys.path.insert(0, str(Path(__file__).parent / "ml"))
from model_utils import MMMModel

# Load data
if not os.path.exists('data/synthetic-marketing-sales.csv'):
    print("Generating synthetic data...")
    import subprocess
    subprocess.run(['python', 'data/generate_data.py'])

df = pd.read_csv('data/synthetic-marketing-sales.csv')
df['date'] = pd.to_datetime(df['date'])
channel_metadata = pd.read_csv('data/synthetic-channel-metadata.csv')

# Load MMM model
if not os.path.exists('ml/mmm_model.joblib'):
    print("Training MMM model...")
    import subprocess
    subprocess.run(['python', 'ml/train_model.py'])

mmm = MMMModel()

# Channel names
spend_cols = [col for col in df.columns if col.endswith('_spend')]
channel_names = [col.replace('_spend', '').replace('_', ' ').title() for col in spend_cols]
channel_map = dict(zip(channel_names, spend_cols))

# App UI
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Overview",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Dashboard Controls"),
                ui.input_date_range(
                    "date_range",
                    "Date Range:",
                    start=df['date'].min(),
                    end=df['date'].max()
                ),
                ui.input_checkbox_group(
                    "channels",
                    "Select Channels:",
                    choices=channel_names,
                    selected=channel_names
                ),
                ui.hr(),
                ui.markdown(
                    """
                    **About This Dashboard**
                    
                    Explore marketing performance, channel contributions, and budget optimization insights.
                    
                    *This project contains synthetic data and analysis created for demonstration purposes only.*
                    """
                ),
                width=300
            ),
            ui.card(
                ui.card_header("Key Performance Metrics"),
                ui.output_ui("kpi_cards")
            ),
            ui.card(
                ui.card_header("Sales & Marketing Spend Over Time"),
                ui.output_ui("sales_trend_plot")
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Channel Spend Distribution"),
                    ui.output_ui("channel_distribution_plot")
                ),
                ui.card(
                    ui.card_header("ROI by Channel"),
                    ui.output_ui("roi_chart")
                )
            )
        )
    ),
    ui.nav_panel(
        "Channel Analysis",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "analysis_channel",
                    "Select Channel:",
                    choices=channel_names,
                    selected=channel_names[0]
                ),
                ui.hr(),
                ui.output_ui("channel_stats"),
                width=300
            ),
            ui.card(
                ui.card_header("Channel Spend vs Sales Contribution"),
                ui.output_ui("channel_scatter_plot")
            ),
            ui.card(
                ui.card_header("Weekly Performance"),
                ui.output_ui("channel_time_series")
            )
        )
    ),
    ui.nav_panel(
        "Budget Optimizer",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Budget Optimization"),
                ui.input_numeric(
                    "total_budget",
                    "Total Weekly Budget ($K):",
                    value=float(df[[col for col in df.columns if col.endswith('_spend')]].sum(axis=1).mean()),
                    min=50,
                    max=500,
                    step=10
                ),
                ui.input_slider(
                    "min_allocation",
                    "Min Channel Allocation (%):",
                    min=0,
                    max=20,
                    value=5,
                    step=1
                ),
                ui.input_slider(
                    "max_allocation",
                    "Max Channel Allocation (%):",
                    min=20,
                    max=100,
                    value=40,
                    step=5
                ),
                ui.input_action_button("optimize", "Optimize Budget", class_="btn-primary"),
                width=300
            ),
            ui.card(
                ui.card_header("Recommended Budget Allocation"),
                ui.output_ui("optimization_results")
            ),
            ui.card(
                ui.card_header("Current vs Recommended Allocation"),
                ui.output_ui("allocation_comparison")
            )
        )
    ),
    ui.nav_panel(
        "Model Insights",
        ui.layout_columns(
            ui.card(
                ui.card_header("Model Performance"),
                ui.output_ui("model_metrics")
            ),
            ui.card(
                ui.card_header("Feature Importance"),
                ui.output_ui("feature_importance_plot")
            )
        ),
        ui.card(
            ui.card_header("Actual vs Predicted Sales"),
            ui.output_ui("predictions_plot")
        )
    ),
    title="Marketing Mix Modeling Dashboard - BrickCraft Studios",
    theme=ui.Theme.from_brand(__file__)
)

# Server logic
def server(input, output, session):
    
    @reactive.calc
    def filtered_data():
        """Filter data based on date range and channel selection"""
        date_min, date_max = input.date_range()
        mask = (df['date'] >= pd.to_datetime(date_min)) & (df['date'] <= pd.to_datetime(date_max))
        return df[mask].copy()
    
    @render.ui
    def kpi_cards():
        """Display key performance indicators"""
        data = filtered_data()
        
        avg_sales = data['sales_revenue'].mean()
        total_spend = data[[col for col in data.columns if col.endswith('_spend')]].sum(axis=1).mean()
        overall_roi = (data[[col for col in data.columns if col.endswith('_contribution')]].sum(axis=1).sum() / 
                       data[[col for col in data.columns if col.endswith('_spend')]].sum(axis=1).sum())
        
        return ui.layout_columns(
            ui.value_box(
                "Average Weekly Sales",
                f"${avg_sales:,.0f}K",
                showcase=None,
                theme="primary"
            ),
            ui.value_box(
                "Average Weekly Spend",
                f"${total_spend:,.0f}K",
                showcase=None,
                theme="info"
            ),
            ui.value_box(
                "Overall Marketing ROI",
                f"{overall_roi:.2f}x",
                showcase=None,
                theme="success"
            )
        )
    
    @render.ui
    def sales_trend_plot():
        """Plot sales and spend trends"""
        data = filtered_data()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Weekly Sales Revenue', 'Total Marketing Spend'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        fig.add_trace(
            go.Scatter(x=data['date'], y=data['sales_revenue'],
                      mode='lines', name='Sales Revenue',
                      line=dict(color='#D01012', width=2),
                      fill='tozeroy'),
            row=1, col=1
        )
        
        total_spend = data[[col for col in data.columns if col.endswith('_spend')]].sum(axis=1)
        fig.add_trace(
            go.Scatter(x=data['date'], y=total_spend,
                      mode='lines', name='Total Marketing Spend',
                      line=dict(color='#0055BF', width=2),
                      fill='tozeroy'),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Revenue ($K)", row=1, col=1)
        fig.update_yaxes(title_text="Spend ($K)", row=2, col=1)
        fig.update_layout(height=600, showlegend=False, hovermode='x unified')
        
        return ui.HTML(fig.to_html())
    
    @render.ui
    def channel_distribution_plot():
        """Show channel spend distribution"""
        data = filtered_data()
        selected = [channel_map[ch] for ch in input.channels() if ch in channel_map]
        
        if not selected:
            return ui.p("No channels selected")
        
        total_by_channel = {col.replace('_spend', '').replace('_', ' ').title(): 
                           data[col].sum() for col in selected}
        
        fig = go.Figure(data=[go.Pie(
            labels=list(total_by_channel.keys()),
            values=list(total_by_channel.values()),
            hole=.3
        )])
        
        fig.update_layout(height=400)
        return ui.HTML(fig.to_html())
    
    @render.ui
    def roi_chart():
        """Display ROI by channel"""
        data = filtered_data()
        selected = [channel_map[ch] for ch in input.channels() if ch in channel_map]
        
        if not selected:
            return ui.p("No channels selected")
        
        roi_data = []
        for spend_col in selected:
            contrib_col = spend_col.replace('_spend', '_contribution')
            if contrib_col in data.columns:
                channel = spend_col.replace('_spend', '').replace('_', ' ').title()
                total_spend = data[spend_col].sum()
                total_contrib = data[contrib_col].sum()
                roi = total_contrib / total_spend if total_spend > 0 else 0
                roi_data.append({'Channel': channel, 'ROI': roi})
        
        roi_df = pd.DataFrame(roi_data).sort_values('ROI', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=roi_df['ROI'],
            y=roi_df['Channel'],
            orientation='h',
            marker_color='#00A651',
            text=[f'{x:.2f}x' for x in roi_df['ROI']],
            textposition='outside'
        ))
        
        fig.update_layout(
            xaxis_title='ROI',
            height=400,
            showlegend=False
        )
        
        return ui.HTML(fig.to_html())
    
    @render.ui
    def channel_stats():
        """Display statistics for selected channel"""
        data = filtered_data()
        channel = input.analysis_channel()
        spend_col = channel_map[channel]
        contrib_col = spend_col.replace('_spend', '_contribution')
        
        avg_spend = data[spend_col].mean()
        total_spend = data[spend_col].sum()
        avg_contrib = data[contrib_col].mean() if contrib_col in data.columns else 0
        total_contrib = data[contrib_col].sum() if contrib_col in data.columns else 0
        roi = total_contrib / total_spend if total_spend > 0 else 0
        
        return ui.markdown(f"""
        ### {channel} Statistics
        
        **Spend:**
        - Average: ${avg_spend:,.2f}K/week
        - Total: ${total_spend:,.2f}K
        
        **Contribution:**
        - Average: ${avg_contrib:,.2f}K/week
        - Total: ${total_contrib:,.2f}K
        
        **ROI:** {roi:.2f}x
        """)
    
    @render.ui
    def channel_scatter_plot():
        """Scatter plot of spend vs contribution"""
        data = filtered_data()
        channel = input.analysis_channel()
        spend_col = channel_map[channel]
        contrib_col = spend_col.replace('_spend', '_contribution')
        
        if contrib_col not in data.columns:
            return ui.p("Contribution data not available")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data[spend_col],
            y=data[contrib_col],
            mode='markers',
            marker=dict(color='#D01012', size=8, opacity=0.6),
            name=channel
        ))
        
        fig.update_layout(
            xaxis_title=f'{channel} Spend ($K)',
            yaxis_title=f'{channel} Contribution ($K)',
            height=400,
            showlegend=False
        )
        
        return ui.HTML(fig.to_html())
    
    @render.ui
    def channel_time_series():
        """Time series of channel spend and contribution"""
        data = filtered_data()
        channel = input.analysis_channel()
        spend_col = channel_map[channel]
        contrib_col = spend_col.replace('_spend', '_contribution')
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=data['date'], y=data[spend_col],
                      mode='lines', name='Spend',
                      line=dict(color='#0055BF', width=2)),
            secondary_y=False
        )
        
        if contrib_col in data.columns:
            fig.add_trace(
                go.Scatter(x=data['date'], y=data[contrib_col],
                          mode='lines', name='Contribution',
                          line=dict(color='#00A651', width=2)),
                secondary_y=True
            )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Spend ($K)", secondary_y=False)
        fig.update_yaxes(title_text="Contribution ($K)", secondary_y=True)
        fig.update_layout(height=400, hovermode='x unified')
        
        return ui.HTML(fig.to_html())
    
    @reactive.calc
    @reactive.event(input.optimize)
    def optimized_budget():
        """Calculate optimized budget allocation"""
        total_budget = input.total_budget()
        min_pct = input.min_allocation() / 100
        max_pct = input.max_allocation() / 100
        
        return mmm.optimize_budget(total_budget, min_pct, max_pct)
    
    @render.ui
    @reactive.event(input.optimize)
    def optimization_results():
        """Display optimization results"""
        result = optimized_budget()
        
        # Create styled table
        result_styled = result.copy()
        result_styled['Budget %'] = result_styled['Budget %'].apply(lambda x: f"{x:.1f}%")
        result_styled['Recommended Spend ($K)'] = result_styled['Recommended Spend ($K)'].apply(lambda x: f"${x:,.2f}")
        
        html_table = result_styled.to_html(index=False, classes='table table-striped')
        
        return ui.HTML(f"""
        <div class="table-responsive">
            {html_table}
        </div>
        <p class="mt-3"><strong>Total Budget: ${input.total_budget():,.2f}K per week</strong></p>
        """)
    
    @render.ui
    @reactive.event(input.optimize)
    def allocation_comparison():
        """Compare current vs recommended allocation"""
        result = optimized_budget()
        data = filtered_data()
        
        # Current allocation
        current_alloc = []
        for _, row in result.iterrows():
            channel = row['Channel']
            spend_col = channel.lower().replace(' ', '_') + '_spend'
            if spend_col in data.columns:
                current_spend = data[spend_col].mean()
                current_alloc.append(current_spend)
            else:
                current_alloc.append(0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current',
            x=result['Channel'],
            y=current_alloc,
            marker_color='#6B6B6B'
        ))
        
        fig.add_trace(go.Bar(
            name='Recommended',
            x=result['Channel'],
            y=result['Recommended Spend ($K)'],
            marker_color='#00A651'
        ))
        
        fig.update_layout(
            barmode='group',
            xaxis_title='Channel',
            yaxis_title='Weekly Spend ($K)',
            height=400
        )
        
        return ui.HTML(fig.to_html())
    
    @render.ui
    def model_metrics():
        """Display model performance metrics"""
        metrics = mmm.get_metrics()
        
        return ui.HTML(f"""
        <div class="row">
            <div class="col-md-6">
                <h5>Training Set</h5>
                <ul>
                    <li><strong>R²:</strong> {metrics['train_r2']:.4f}</li>
                    <li><strong>MAE:</strong> ${metrics['train_mae']:.2f}K</li>
                    <li><strong>RMSE:</strong> ${metrics['train_rmse']:.2f}K</li>
                    <li><strong>MAPE:</strong> {metrics['train_mape']:.2f}%</li>
                </ul>
            </div>
            <div class="col-md-6">
                <h5>Test Set</h5>
                <ul>
                    <li><strong>R²:</strong> {metrics['test_r2']:.4f}</li>
                    <li><strong>MAE:</strong> ${metrics['test_mae']:.2f}K</li>
                    <li><strong>RMSE:</strong> ${metrics['test_rmse']:.2f}K</li>
                    <li><strong>MAPE:</strong> {metrics['test_mape']:.2f}%</li>
                </ul>
            </div>
        </div>
        """)
    
    @render.ui
    def feature_importance_plot():
        """Display feature importance"""
        importance = mmm.get_feature_importance()
        importance = importance.head(10)
        
        fig = go.Figure(go.Bar(
            x=importance['Coefficient'],
            y=importance['Feature'],
            orientation='h',
            marker_color=['#00A651' if x > 0 else '#D01012' for x in importance['Coefficient']],
            text=[f'{x:.2f}' for x in importance['Coefficient']],
            textposition='outside'
        ))
        
        fig.update_layout(
            xaxis_title='Coefficient',
            yaxis_title='Feature',
            height=500,
            showlegend=False
        )
        
        return ui.HTML(fig.to_html())
    
    @render.ui
    def predictions_plot():
        """Plot actual vs predicted values"""
        if not os.path.exists('ml/test_predictions.csv'):
            return ui.p("Prediction data not available")
        
        pred_df = pd.read_csv('ml/test_predictions.csv')
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pred_df['date'],
            y=pred_df['actual'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#D01012', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=pred_df['date'],
            y=pred_df['predicted'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#0055BF', width=2, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Sales Revenue ($K)',
            height=500,
            hovermode='x unified'
        )
        
        return ui.HTML(fig.to_html())


app = App(app_ui, server)
