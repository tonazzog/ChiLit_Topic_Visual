import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import pandas as pd
import json
import matplotlib as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

chunk_size = 200
data_folder="./data/"

# ChiLit books
df_metadata = pd.read_csv(f"{data_folder}ChiLit_metadata.csv", encoding="utf-8")
df_authors = pd.read_csv(f"{data_folder}ChiLit_authors.csv", encoding="utf-8")
df_chilit = pd.read_csv(f"{data_folder}ChiLit_Chunks_{chunk_size}.csv")
df_chilit = df_chilit.fillna("")

# ProdLDA model optimized by OCTIS
final_model = pickle.load(open(f"{data_folder}Octis_ProdLDA_output.pkl", "rb"))

# Topic Labesl
with open(f"{data_folder}Octis_ProdLDA_Topic_Labels.json", 'r') as file:
  labels = json.load(file)
topic_labels = [value['primary_label'] for value in labels.values()]

# Add document original information (book, chapter)
n_topics = len(final_model['topics'])
df_topics = pd.DataFrame(final_model['topic-document-matrix'].T, columns=topic_labels)
df_topics['book_id'] = df_chilit['book_id'].to_list()
df_topics['chapter_num'] = df_chilit['chapter_num'].to_list()

# Aggreagate topics by book
agg_df = df_topics.drop(['chapter_num'], axis=1).groupby("book_id").mean()

# Color map
colormap = plt.colormaps['tab20'].colors  # can be 'hsv', 'tab20', 'nipy_spectral', etc.
color_sequence = [mcolors.to_hex(colormap[i]) for i in range(n_topics)]

# Create consistent color mapping for all topics
topic_color_map = {topic: color_sequence[i] for i, topic in enumerate(topic_labels)}

def create_topic_relevance_dash_app(df, title="Topic Relevance Explorer", color_sequence=px.colors.qualitative.Set3):
    """
    Create a Dash app with an interactive stacked bar chart.
    
    Parameters:
    - df: DataFrame with books as index and topics as columns
    - title: Title for the chart
    - color_sequence: Color sequence for the chart
    
    Returns:
    - Dash app instance
    """
    
    # Reset index to get book_id as a column
    df_reset = df.reset_index()
    topics = df.columns.tolist()
    
    # Initialize the Dash app
    app = dash.Dash(__name__)
    
    # Define the layout
    app.layout = html.Div([
        html.Div([
            html.H2(title, style={
                'textAlign': 'center',
                'color': '#333',
                'marginBottom': '30px'
            }),
            
            html.Div([
                html.Div([
                    html.Label('Select Topics:', style={
                        'fontWeight': 'bold',
                        'marginRight': '15px'
                    }),
                    html.Button('Select All', id='select-all-btn', 
                               style={
                                   'backgroundColor': '#007bff',
                                   'color': 'white',
                                   'border': 'none',
                                   'padding': '8px 16px',
                                   'borderRadius': '4px',
                                   'cursor': 'pointer',
                                   'marginLeft': '10px',
                                   'marginRight': '10px'
                               }),
                    html.Button('Clear All', id='clear-all-btn',
                               style={
                                   'backgroundColor': '#007bff',
                                   'color': 'white',
                                   'border': 'none',
                                   'padding': '8px 16px',
                                   'borderRadius': '4px',
                                   'cursor': 'pointer'
                               })
                ], style={'marginBottom': '15px'}),
                
                dcc.Checklist(
                    id='topic-selector',
                    options=[{'label': topic, 'value': topic} for topic in topics],
                    value=topics,  # All topics selected by default
                    inline=True,
                    style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '15px'}
                )
            ], className='controls', style={
                'marginBottom': '20px',
                'padding': '15px',
                'backgroundColor': '#f9f9f9',
                'borderRadius': '5px',
                'border': '1px solid #ddd'
            }),
            
            dcc.Graph(id='topic-chart', style={'height': '1400px'})
            
        ], className='container', style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'margin': '20px'
        })
    ], style={
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#f5f5f5',
        'minHeight': '100vh'
    })
    
    # Callback for select/clear all buttons
    @app.callback(
        Output('topic-selector', 'value'),
        [Input('select-all-btn', 'n_clicks'),
         Input('clear-all-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def update_checklist(select_clicks, clear_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            return topics
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'select-all-btn':
            return topics
        elif button_id == 'clear-all-btn':
            return []
        
        return topics
    
    # Callback for updating the chart
    @app.callback(
        Output('topic-chart', 'figure'),
        [Input('topic-selector', 'value')]
    )
    def update_chart(selected_topics):
        if not selected_topics:
            # Return empty chart with message
            fig = go.Figure()
            fig.update_layout(
                title="Please select at least one topic",
                xaxis_title="Relevance",
                yaxis_title="Book ID",
                height=400,
                annotations=[{
                    'text': 'No topics selected',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'middle',
                    'showarrow': False,
                    'font': {'size': 20, 'color': '#888'}
                }]
            )
            return fig
        
        selected_topics = [t for t in topics if t in selected_topics]

        # Filter dataframe for selected topics
        df_filtered = df_reset[['book_id'] + selected_topics].copy()
        
        # Convert to long format
        df_long = df_filtered.melt(
            id_vars="book_id",
            value_vars=selected_topics,
            var_name="Topic",
            value_name="Relevance"
        )
        
        # Calculate total relevance per book for sorting
        book_totals = (
            df_long.groupby("book_id")["Relevance"]
            .sum()
            .round(3)
            .reset_index()  # turn it into a DataFrame
            .sort_values(by=["Relevance", "book_id"], ascending=[False, True])
            .set_index("book_id")["Relevance"]
        )
        
        # Create the chart
        fig = px.bar(
            df_long,
            x="Relevance",
            y="book_id",
            color="Topic",
            orientation="h",
            title=f"Books sorted by total relevance",
            category_orders={"book_id": book_totals.index.tolist()},
            color_discrete_sequence=[topic_color_map[t] for t in selected_topics]
        )
        
        fig.update_layout(
            barmode="stack",
            yaxis_title="Book ID",
            xaxis_title="Relevance",
            height=1400,
            margin=dict(l=150, r=50, t=100, b=50),
            hovermode='closest',
            title_x=0.5  # center the title
        )
        
        # Customize hover template
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>%{fullData.name}: %{x:.2f}<br><extra></extra>'
        )
        
        return fig
    
    return app

# Example usage:
def run_topic_relevance_app(df, host='127.0.0.1', port=8050, debug=True):
    """
    Run the Dash app for topic relevance visualization.
    
    Parameters:
    - df: DataFrame with books as index and topics as columns
    - host: Host address (default: '127.0.0.1')
    - port: Port number (default: 8050)
    - debug: Enable debug mode (default: True)
    """
    app = create_topic_relevance_dash_app(df, color_sequence=color_sequence)
    app.run(host=host, port=port, debug=debug)

# To use with your data:
# app = create_topic_relevance_dash_app(agg_df)
# app.run(debug=True)

# Or use the convenience function:
run_topic_relevance_app(agg_df)