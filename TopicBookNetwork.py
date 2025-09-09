import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import networkx as nx
import numpy as np
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
book_topic_df = df_topics.drop(['chapter_num'], axis=1).groupby("book_id").mean()
# Initialize the Dash app
app = dash.Dash(__name__)



def create_network_graph(book_topic_df, n_top=5, min_weight=0.0, selected_books=None, selected_topics=None):
    """Create NetworkX graph from book-topic data with filters"""
    
    # Filter data if selections are provided
    filtered_df = book_topic_df.copy()
    
    if selected_books:
        filtered_df = filtered_df.loc[selected_books]
    
    if selected_topics:
        filtered_df = filtered_df[selected_topics]
    
    books = list(filtered_df.index)
    topics = list(filtered_df.columns)
    
    # Initialize graph
    B = nx.Graph()
    
    # Add book nodes
    B.add_nodes_from(books, bipartite="books", node_type='books')
    
    # Add topic nodes  
    B.add_nodes_from(topics, bipartite="topics", node_type='topics')
    
    # Add edges (book → top n topics) with weight filter
    for book, row in filtered_df.iterrows():
        top_topics = row.nlargest(n_top).index
        for topic in top_topics:
            weight = row[topic]
            if weight >= min_weight:
                B.add_edge(book, topic, weight=weight)
    
    return B, books, topics, filtered_df

def create_plotly_figure(B, books, topics, book_topic_df):
    """Convert NetworkX graph to Plotly figure"""
    
    if len(B.nodes()) == 0:
        # Return empty figure if no nodes
        return go.Figure().add_annotation(
            text="No data matches current filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
    
    # Compute layout
    pos = nx.spring_layout(B, seed = 42)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in B.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = B[edge[0]][edge[1]]['weight']
        edge_info.append(f"{edge[0]} ↔ {edge[1]}<br>Weight: {weight:.3f}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Prepare node traces for books
    book_x = [pos[node][0] for node in books if node in pos]
    book_y = [pos[node][1] for node in books if node in pos]
    book_text = [node for node in books if node in pos]
    book_sizes = [max(20, 20 + 5*B.degree(book)) if book in B else 20 for book in books if book in pos]
    book_hover = [f"Book: {book}<br>Connections: {B.degree(book) if book in B else 0}" 
                  for book in books if book in pos]
    
    book_trace = go.Scatter(
        x=book_x, y=book_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=book_hover,
        text=book_text,
        textposition="middle center",
        marker=dict(
            size=book_sizes,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        name='Books'
    )
    
    # Prepare node traces for topics
    topic_x = [pos[node][0] for node in topics if node in pos]
    topic_y = [pos[node][1] for node in topics if node in pos]
    topic_text = [node for node in topics if node in pos]
    topic_sizes = [max(20, 20 + 500 * book_topic_df[topic].mean()) if topic in book_topic_df.columns else 20 
                   for topic in topics if topic in pos]
    topic_hover = [f"Topic: {topic}<br>Avg Relevance: {book_topic_df[topic].mean():.3f}<br>Connections: {B.degree(topic) if topic in B else 0}" 
                   for topic in topics if topic in pos]
    
    topic_trace = go.Scatter(
        x=topic_x, y=topic_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=topic_hover,
        text=topic_text,
        textposition="middle center",
        marker=dict(
            size=topic_sizes,
            color='lightgreen',
            line=dict(width=2, color='darkgreen')
        ),
        name='Topics'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, book_trace, topic_trace],
                   layout=go.Layout(
                       title='Interactive Book-Topic Network',
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Hover over nodes for details. Use filters to explore the network.",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='#888', size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white'
                   ))
    
    return fig

# Define the app layout
app.layout = html.Div([
    html.H1("Interactive Book-Topic Network Explorer", style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Number of Top Topics per Book:", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='n-top-slider',
                min=1, max=10, value=5, step=1,
                marks={i: str(i) for i in range(1, 11)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '5%'}),
        
        html.Div([
            html.Label("Minimum Edge Weight:", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='weight-slider',
                min=0, max=1, value=0, step=0.05,
                marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '5%'}),
        
        html.Div([
            html.Button('Reset Filters', id='reset-button', n_clicks=0,
                       style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none',
                              'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'center'})
        
    ], style={'marginBottom': 30, 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    
    # Dropdown filters
    html.Div([
        html.Div([
            html.Label("Select Books:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='book-dropdown',
                options=[{'label': book, 'value': book} for book in book_topic_df.index],
                value=list(book_topic_df.index),
                multi=True,
                placeholder="Select books to include..."
            )
        ], style={'width': '62%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.Label("Select Topics:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='topic-dropdown',
                options=[{'label': topic, 'value': topic} for topic in book_topic_df.columns],
                value=list(book_topic_df.columns),
                multi=True,
                placeholder="Select topics to include..."
            )
        ], style={'width': '36%', 'display': 'inline-block'})
        
    ], style={'marginBottom': 30}),
    
    # Network graph
    dcc.Graph(id='network-graph', style={'height': '100vh'}),
    
    # Statistics panel
    html.Div(id='stats-panel', style={'marginTop': 20, 'padding': '15px', 
                                     'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
])

# Callbacks
@app.callback(
    [Output('network-graph', 'figure'),
     Output('stats-panel', 'children')],
    [Input('n-top-slider', 'value'),
     Input('weight-slider', 'value'),
     Input('book-dropdown', 'value'),
     Input('topic-dropdown', 'value')]
)
def update_graph(n_top, min_weight, selected_books, selected_topics):
    # Create network graph
    B, books, topics, filtered_df = create_network_graph(
        book_topic_df, n_top, min_weight, selected_books, selected_topics
    )
    
    # Create plotly figure
    fig = create_plotly_figure(B, books, topics, filtered_df)
    
    # Create statistics
    stats = html.Div([
        html.H3("Network Statistics", style={'marginBottom': 15}),
        html.Div([
            html.Span(f"Books: {len(books)}", style={'marginRight': 30, 'fontWeight': 'bold'}),
            html.Span(f"Topics: {len(topics)}", style={'marginRight': 30, 'fontWeight': 'bold'}),
            html.Span(f"Edges: {B.number_of_edges()}", style={'marginRight': 30, 'fontWeight': 'bold'}),
            html.Span(f"Avg Edge Weight: {np.mean([B[u][v]['weight'] for u,v in B.edges()]) if B.edges() else 0:.3f}", 
                     style={'fontWeight': 'bold'})
        ])
    ])
    
    return fig, stats

@app.callback(
    [Output('book-dropdown', 'value'),
     Output('topic-dropdown', 'value'),
     Output('n-top-slider', 'value'),
     Output('weight-slider', 'value')],
    [Input('reset-button', 'n_clicks')]
)
def reset_filters(n_clicks):
    if n_clicks > 0:
        return list(book_topic_df.index), list(book_topic_df.columns), 5, 0
    return dash.no_update

if __name__ == '__main__':
    app.run(debug=True)