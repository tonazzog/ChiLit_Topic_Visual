import dash
from dash import dcc, html, Input, Output, callback, State, ctx
import plotly.graph_objects as go
import pandas as pd
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
df_visual = df_topics.reset_index().rename(columns ={'index':'chunk_num'})
topic_cols = df_visual.columns.tolist()[1:17]
df_visual = df_visual[['book_id','chapter_num','chunk_num'] + topic_cols].reset_index()
df_visual = pd.merge(df_visual, df_metadata, left_on='book_id',right_on='shorttitle')
df_visual = pd.merge(df_visual, df_authors, on='author')
agg_df = df_visual[['author'] + topic_cols]
agg_df = agg_df.groupby('author').mean()


# Color map
colormap = plt.colormaps['tab20'].colors  # can be 'hsv', 'tab20', 'nipy_spectral', etc.
color_sequence = [mcolors.to_hex(colormap[i]) for i in range(n_topics)]


title="Topic Relevance Heatmap"
row_height=30

    
# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

all_authors = list(agg_df.index)
all_topics = list(agg_df.columns)

# Define the app layout
app.layout = html.Div([
    html.Div([
        html.H2(title, style={'textAlign': 'center', 'color': '#333', 'marginBottom': '30px'}),
        
        # Author selection controls
        html.Div([
            html.H3('Select Authors to Display:', style={'marginBottom': '15px'}),
            
            html.Div([
                html.Button('Select All', id='select-all-authors-btn', n_clicks=0,
                            className='control-btn'),
                html.Button('Deselect All', id='deselect-all-authors-btn', n_clicks=0,
                            className='control-btn'),
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Button(
                    author,
                    id={'type': 'author-btn', 'index': author},
                    className='author-btn selected',
                    n_clicks=0,
                    style={'margin': '3px', 'padding': '5px 10px', 'border': '1px solid #45a049',
                            'borderRadius': '3px', 'cursor': 'pointer', 'display': 'inline-block',
                            'fontSize': '12px', 'backgroundColor': '#4CAF50', 'color': 'white'}
                ) for author in all_authors
            ], id='author-buttons-container')
            
        ], className='author-controls'),
        
        # Store selected authors
        dcc.Store(id='selected-authors-store', data=all_authors),
        
        # Heatmap
        dcc.Graph(id='heatmap-chart', style={'width': '100%'})
        
    ], className='container')
], className='main-wrapper')

# Updated callback using the newer ctx approach
@app.callback(
    Output('selected-authors-store', 'data'),
    [Input('select-all-authors-btn', 'n_clicks'),
        Input('deselect-all-authors-btn', 'n_clicks'),
        Input({'type': 'author-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('selected-authors-store', 'data')],
    prevent_initial_call=True
)
def update_selected_authors(select_all_clicks, deselect_all_clicks, individual_clicks, current_selected):
    if not ctx.triggered:
        return current_selected or all_authors
    
    triggered_id = ctx.triggered_id
    
    # Handle select all
    if triggered_id == 'select-all-authors-btn':
        return all_authors
    
    # Handle deselect all
    elif triggered_id == 'deselect-all-authors-btn':
        return []
    
    # Handle individual author button clicks
    elif isinstance(triggered_id, dict) and triggered_id.get('type') == 'author-btn':
        author = triggered_id.get('index')
        if author:
            current_selected = current_selected or []
            new_selected = current_selected.copy()
            
            if author in new_selected:
                new_selected.remove(author)
            else:
                new_selected.append(author)
            
            return new_selected
    
    return current_selected or all_authors

# Single callback to update button styles based on selected authors
@app.callback(
    [Output({'type': 'author-btn', 'index': dash.dependencies.ALL}, 'className'),
        Output({'type': 'author-btn', 'index': dash.dependencies.ALL}, 'style')],
    [Input('selected-authors-store', 'data')],
    [State({'type': 'author-btn', 'index': dash.dependencies.ALL}, 'id')]
)
def update_button_styles(selected_authors, button_ids):
    classNames = []
    styles = []
    
    # Handle case where selected_authors might be None
    selected_authors = selected_authors or []
    
    for button_id in button_ids:
        author = button_id['index']
        if author in selected_authors:
            classNames.append('author-btn selected')
            styles.append({
                'margin': '3px', 'padding': '5px 10px', 'border': '1px solid #45a049',
                'borderRadius': '3px', 'cursor': 'pointer', 'display': 'inline-block',
                'fontSize': '12px', 'backgroundColor': '#4CAF50', 'color': 'white'
            })
        else:
            classNames.append('author-btn unselected')
            styles.append({
                'margin': '3px', 'padding': '5px 10px', 'border': '1px solid #495057',
                'borderRadius': '3px', 'cursor': 'pointer', 'display': 'inline-block',
                'fontSize': '12px', 'backgroundColor': '#adb5bd', 'color': 'white'
            })
    
    return classNames, styles

# Main callback for updating the heatmap
@app.callback(
    Output('heatmap-chart', 'figure'),
    [Input('selected-authors-store', 'data')]
)
def update_heatmap(selected_authors):
    # Handle case where selected_authors might be None or empty
    if not selected_authors:
        # Return empty heatmap with message
        fig = go.Figure()
        fig.add_annotation(
            text="Please select at least one author to display the heatmap",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title=title,
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    # Filter the dataframe to selected authors
    filtered_df = agg_df.loc[selected_authors]
    
    # Create the heatmap
    fig = go.Figure()
    
    fig.add_trace(
        go.Heatmap(
            z=filtered_df.values,
            x=filtered_df.columns,
            y=filtered_df.index,
            colorscale='YlGnBu',
            hoverongaps=False,
            hovertemplate='Document: %{y}<br>Topic: %{x}<br>Relevance: %{z:.4f}<extra></extra>',
            colorbar=dict(title="Relevance Score"),
            name="Selected Authors"
        )
    )
    
    # Calculate height based on number of selected authors
    chart_height = row_height * len(selected_authors) + 300
    
    fig.update_layout(
        title=dict(
            text=f"{title} ({len(selected_authors)} authors selected)",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Topics",
        yaxis_title="Authors",
        height=chart_height,
        width=1400,
        font=dict(size=12),
        xaxis=dict(
            tickangle=45,
            side='bottom'
        ),
        yaxis=dict(
            tickangle=0,
            autorange='reversed'
        ),
        margin=dict(t=100, l=150, r=50, b=150)
    )
    
    return fig

# Add CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .main-wrapper {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .author-controls {
                margin: 20px 0;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            
            .author-btn {
                transition: all 0.2s;
            }
            
            .author-btn:hover {
                opacity: 0.8;
            }
            
            .control-btn {
                margin: 5px;
                padding: 8px 15px;
                border: 1px solid #2196F3;
                border-radius: 4px;
                background-color: #2196F3;
                color: white;
                cursor: pointer;
                font-weight: bold;
            }
            
            .control-btn:hover {
                background-color: #1976D2;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
    
if __name__ == '__main__':
    app.run(debug=True)