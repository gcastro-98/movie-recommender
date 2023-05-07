"""
Implementation of some plotly graphics to enhance the web app aesthetics.
"""

import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objs


def perform_bubble_plot(
        genre_grouped_df: pd.DataFrame) -> plotly.graph_objs.Figure:
    genre_grouped_df = genre_grouped_df.drop(
        labels=['Genres'], axis=1).reset_index()
    genre_grouped_df['Rating'] = genre_grouped_df['Rating'].replace(
        'Unknown', np.nan).dropna(axis=0).astype(float)
    fig = px.scatter(genre_grouped_df, x='Year', y='Rating',
                     color="Genres", hover_data=['DisplayTitle'])

    fig.update_layout(title="Movie release year vs. rating",
                      xaxis_title="Release year",
                      yaxis_title="Rating")
    return fig


def perform_bar_plot(genre_grouped_df: pd.DataFrame) -> plotly.graph_objs.Figure:
    genre_grouped_df = genre_grouped_df.drop(
        labels=['Genres'], axis=1).reset_index()
    genre_grouped_df['Rating'] = genre_grouped_df['Rating'].replace(
        'Unknown', np.nan).dropna(axis=0).astype(float)
    genre_df = genre_grouped_df['Genres'].value_counts().reset_index()
    genre_df.columns = ['Genres', 'Count']
    fig = px.bar(genre_df, x='Genres', y='Count', color_discrete_sequence=["#003366"])

    fig.update_layout(title="Movies per genre",
                      xaxis_title="Genre",
                      yaxis_title="Number of movies")
    return fig


if __name__ == '__main__':
    _df: pd.DataFrame = pd.read_csv(
        '/home/gcastro/Downloads/agile/ml-1m/curated/genre-grouped.csv')
