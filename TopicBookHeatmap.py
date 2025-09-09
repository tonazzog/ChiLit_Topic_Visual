import dash
from dash import dcc, html, Input, Output, callback, State
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

# Aggreagate topics by book
agg_df = df_topics.groupby("book_id").mean()

# Color map
colormap = plt.colormaps['tab20'].colors  # can be 'hsv', 'tab20', 'nipy_spectral', etc.
color_sequence = [mcolors.to_hex(colormap[i]) for i in range(n_topics)]

def create_topic_heatmap_app(agg_df, title="Topic Relevance Heatmap", row_height=30):
    """
    Create a Dash app with an interactive heatmap and individual book selection.
    
    Parameters:
    - agg_df: DataFrame with books as index and topics as columns
    - title: Title for the app
    - row_height: Height per row in pixels
    
    Returns:
    - Dash app instance
    """
    
    # Initialize the Dash app
    app = dash.Dash(__name__)
    server = app.server

    all_books = list(agg_df.index)
    all_topics = list(agg_df.columns)
    
    # Define the app layout
    app.layout = html.Div([
        html.Div([
            html.H2(title, style={'textAlign': 'center', 'color': '#333', 'marginBottom': '30px'}),
            
            # Book selection controls
            html.Div([
                html.H3('Select Books to Display:', style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Button('Select All', id='select-all-books-btn', n_clicks=0,
                              className='control-btn'),
                    html.Button('Deselect All', id='deselect-all-books-btn', n_clicks=0,
                              className='control-btn'),
                ], style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Button(
                        book,
                        id={'type': 'book-btn', 'index': book},
                        className='book-btn selected',
                        n_clicks=0,
                        style={'margin': '3px', 'padding': '5px 10px', 'border': '1px solid #45a049',
                               'borderRadius': '3px', 'cursor': 'pointer', 'display': 'inline-block',
                               'fontSize': '12px', 'backgroundColor': '#4CAF50', 'color': 'white'}
                    ) for book in all_books
                ], id='book-buttons-container')
                
            ], className='book-controls'),
            
            # Store selected books
            dcc.Store(id='selected-books-store', data=all_books),
            
            # Heatmap
            dcc.Graph(id='heatmap-chart', style={'width': '100%'})
            
        ], className='container')
    ], className='main-wrapper')
    
    # Combined callback to handle all book selection changes and update store
    @app.callback(
        Output('selected-books-store', 'data'),
        [Input('select-all-books-btn', 'n_clicks'),
         Input('deselect-all-books-btn', 'n_clicks'),
         Input({'type': 'book-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
        [State('selected-books-store', 'data'),
         State({'type': 'book-btn', 'index': dash.dependencies.ALL}, 'id')],
        prevent_initial_call=True
    )
    def update_selected_books(select_all_clicks, deselect_all_clicks, individual_clicks, current_selected, button_ids):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_selected
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id == 'select-all-books-btn':
            return all_books
        elif triggered_id == 'deselect-all-books-btn':
            return []
        else:
            # Individual book button was clicked
            try:
                triggered_dict = eval(triggered_id)  # Convert string back to dict
                book = triggered_dict['index']
                
                new_selected = current_selected.copy()
                if book in current_selected:
                    new_selected.remove(book)
                else:
                    new_selected.append(book)
                
                return new_selected
            except:
                return current_selected
    
    # Single callback to update button styles based on selected books
    @app.callback(
        [Output({'type': 'book-btn', 'index': dash.dependencies.ALL}, 'className'),
         Output({'type': 'book-btn', 'index': dash.dependencies.ALL}, 'style')],
        [Input('selected-books-store', 'data')],
        [State({'type': 'book-btn', 'index': dash.dependencies.ALL}, 'id')]
    )
    def update_button_styles(selected_books, button_ids):
        classNames = []
        styles = []
        
        for button_id in button_ids:
            book = button_id['index']
            if book in selected_books:
                classNames.append('book-btn selected')
                styles.append({
                    'margin': '3px', 'padding': '5px 10px', 'border': '1px solid #45a049',
                    'borderRadius': '3px', 'cursor': 'pointer', 'display': 'inline-block',
                    'fontSize': '12px', 'backgroundColor': '#4CAF50', 'color': 'white'
                })
            else:
                classNames.append('book-btn unselected')
                styles.append({
                    'margin': '3px', 'padding': '5px 10px', 'border': '1px solid #495057',
                    'borderRadius': '3px', 'cursor': 'pointer', 'display': 'inline-block',
                    'fontSize': '12px', 'backgroundColor': '#adb5bd', 'color': 'white'
                })
        
        return classNames, styles
    
    # Main callback for updating the heatmap
    @app.callback(
        Output('heatmap-chart', 'figure'),
        [Input('selected-books-store', 'data')]
    )
    def update_heatmap(selected_books):
        if not selected_books:
            # Return empty heatmap with message
            fig = go.Figure()
            fig.add_annotation(
                text="Please select at least one book to display the heatmap",
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
        
        # Filter the dataframe to selected books
        filtered_df = agg_df.loc[selected_books]
        
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
                name="Selected Books"
            )
        )
        
        # Calculate height based on number of selected books
        chart_height = row_height * len(selected_books) + 300
        
        fig.update_layout(
            title=dict(
                text=f"{title} ({len(selected_books)} books selected)",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Topics",
            yaxis_title="Books",
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
                
                .book-controls {
                    margin: 20px 0;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: #f9f9f9;
                }
                
                .book-btn {
                    transition: all 0.2s;
                }
                
                .book-btn:hover {
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
    
    return app

# Alternative: Function to run the app directly
def run_topic_heatmap_app(agg_df, title="Topic Relevance Heatmap", row_height=30,
                         debug=True, port=8050, host='127.0.0.1'):
    """
    Create and run a Dash app with an interactive heatmap.
    
    Parameters:
    - agg_df: DataFrame with books as index and topics as columns
    - title: Title for the app
    - row_height: Height per row in pixels
    - debug: Whether to run in debug mode
    - port: Port to run the app on
    - host: Host to run the app on
    """
    app = create_topic_heatmap_app(agg_df, title, row_height)
    app.run(debug=debug, port=port, host=host)


app = create_topic_heatmap_app(agg_df)

if __name__ == '__main__':
    app.run(debug=True)

# To use the direct run version:
#run_topic_heatmap_app(agg_df)