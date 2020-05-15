import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from utils import Header, make_dash_table

import pandas as pd
import pathlib

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()



df_test = pd.read_csv(DATA_PATH.joinpath("df_test.csv"))
df = pd.read_csv(DATA_PATH.joinpath("df_test1.csv"))
df_test2 = pd.read_csv(DATA_PATH.joinpath("df_test2.csv"))


def create_layout(app):
    # Page layouts
    return html.Div(
        [
            html.Div([Header(app)]),
            # page 1
            html.Div(
                [

                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("Result Summary"),
                                    html.Br([]),
                                    html.P(
                                        "\
                                    This program trains three different Reinforcement Learning models to optimize \
                                    customer interaction based on your historical customer data. The comparison \
                                    between different methods is shown in this page to give you a brief overview. \
                                    The details of each model can be viewed in their sub pages. You can choose the\
                                    most suitable model for your future business.",
                                        style={"color": "#ffffff"},
                                        className="row",
                                    ),
                                ],
                                className="product",
                            )
                        ],
                        className="row",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("Overall performance comparison", className="subtitle padded"),
                                    html.Br([]),
                                    dcc.Graph(
                                        id="graph-6",
                                        figure={
                                            "data": [
                                                go.Bar(
                                                    x=df.columns,
                                                    y=[df['Baseline'][0]] * len(df.columns),
                                                    marker={"color": "#97151c"},
                                                    name="A",
                                                ),
                                                go.Bar(
                                                    x=df.drop(columns=['Baseline']).columns,
                                                    y=df.drop(columns=['Baseline']).iloc[0] - df['Baseline'][0],
                                                    marker={"color": " #dddddd"},
                                                    name="B",
                                                ),
                                            ],
                                            "layout": go.Layout(
                                                annotations=[
                                                    {
                                                        "x": 0,
                                                        "y": 24,
                                                        "font": {
                                                            "color": "#7a7a7a",
                                                            "family": "Arial sans serif",
                                                            "size": 8,
                                                        },
                                                        "showarrow": False,
                                                        "text": "$" + str(df['Baseline'][0]),
                                                        "xref": "x",
                                                        "yref": "y",
                                                    },
                                                    {
                                                        "x": 1,
                                                        "y": 24,
                                                        "font": {
                                                            "color": "#7a7a7a",
                                                            "family": "Arial sans serif",
                                                            "size": 8,
                                                        },
                                                        "showarrow": False,
                                                        "text": "$" + str(df['Direct RL'][0]),
                                                        "xref": "x",
                                                        "yref": "y",
                                                    },
                                                    {
                                                        "x": 2,
                                                        "y": 24,
                                                        "font": {
                                                            "color": "#7a7a7a",
                                                            "family": "Arial sans serif",
                                                            "size": 8,
                                                        },
                                                        "showarrow": False,
                                                        "text": "$" + str(df['Indirect RL'][0]),
                                                        "xref": "x",
                                                        "yref": "y",
                                                    },
                                                    {
                                                        "x": 3,
                                                        "y": 24,
                                                        "font": {
                                                            "color": "#7a7a7a",
                                                            "family": "Arial sans serif",
                                                            "size": 8,
                                                        },
                                                        "showarrow": False,
                                                        "text": "$" + str(df['Semidirect RL'][0]),
                                                        "xref": "x",
                                                        "yref": "y",
                                                    },
                                                    {
                                                        "x": 1,
                                                        "y": 55,
                                                        "font": {
                                                            "color": "#97151c",
                                                            "family": "Arial sans serif",
                                                            "size": 8,
                                                        },
                                                        "showarrow": False,
                                                        "text": "$" + str(
                                                            (df.drop(columns=['Baseline']).iloc[0] - df['Baseline'][0])[
                                                                0]) + " more",
                                                        "xref": "x",
                                                        "yref": "y",
                                                    },
                                                    {
                                                        "x": 2,
                                                        "y": 55,
                                                        "font": {
                                                            "color": "#97151c",
                                                            "family": "Arial sans serif",
                                                            "size": 8,
                                                        },
                                                        "showarrow": False,
                                                        "text": "$" + str(
                                                            (df.drop(columns=['Baseline']).iloc[0] - df['Baseline'][0])[
                                                                1]) + " more",
                                                        "xref": "x",
                                                        "yref": "y",
                                                    },
                                                    {
                                                        "x": 3,
                                                        "y": 55,
                                                        "font": {
                                                            "color": "#97151c",
                                                            "family": "Arial sans serif",
                                                            "size": 8,
                                                        },
                                                        "showarrow": False,
                                                        "text": "$" + str(
                                                            (df.drop(columns=['Baseline']).iloc[0] - df['Baseline'][0])[
                                                                2]) + " more",
                                                        "xref": "x",
                                                        "yref": "y",
                                                    },
                                                ],
                                                autosize=True,
                                                height=520,
                                                width=640,
                                                bargap=0.4,
                                                barmode="stack",
                                                hovermode="closest",
                                                margin={
                                                    "r": 40,
                                                    "t": 20,
                                                    "b": 20,
                                                    "l": 40,
                                                },
                                                showlegend=False,
                                                title="",
                                                xaxis={
                                                    "autorange": True,
                                                    "range": [-0.5, 1.5],
                                                    "showline": True,
                                                    "tickfont": {
                                                        "family": "Arial sans serif",
                                                        "size": 8,
                                                    },
                                                    "title": "",
                                                    "type": "category",
                                                    "zeroline": False,
                                                },
                                                yaxis={
                                                    "autorange": False,
                                                    "mirror": False,
                                                    "nticks": 3,
                                                    "range": [0, 150],
                                                    "showgrid": True,
                                                    "showline": True,
                                                    "tickfont": {
                                                        "family": "Arial sans serif",
                                                        "size": 10,
                                                    },
                                                    "tickprefix": "$",
                                                    "title": "",
                                                    "type": "linear",
                                                    "zeroline": False,
                                                },
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("Performance over time", className="subtitle padded"),
                                    dcc.Graph(
                                        id="graph-4",
                                        figure={
                                            "data": [
                                                go.Scatter(
                                                    x=df_test["Episode"],
                                                    y=df_test["Baseline"],
                                                    line={"color": "#97151c"},
                                                    mode="lines",
                                                    name="Baseline",
                                                ),
                                                go.Scatter(
                                                    x=df_test["Episode"],
                                                    y=df_test[
                                                        "Direct RL"
                                                    ],
                                                    line={"color": "#b5b5b5"},
                                                    mode="lines",
                                                    name="Direct RL",
                                                ),
                                                go.Scatter(
                                                    x=df_test["Episode"],
                                                    y=df_test[
                                                        "Indirect RL"
                                                    ],
                                                    line={"color": "#b5b5b5"},
                                                    mode="lines",
                                                    name="Indirect RL",
                                                ),
                                                go.Scatter(
                                                    x=df_test["Episode"],
                                                    y=df_test[
                                                        "Semidirect RL"
                                                    ],
                                                    line={"color": "#b5b5b5"},
                                                    mode="lines",
                                                    name="Semidirect RL",
                                                ),
                                            ],
                                            "layout": go.Layout(
                                                autosize=True,
                                                width=700,
                                                height=200,
                                                font={"family": "Raleway", "size": 10},
                                                margin={
                                                    "r": 30,
                                                    "t": 30,
                                                    "b": 30,
                                                    "l": 30,
                                                },
                                                showlegend=True,
                                                titlefont={
                                                    "family": "Raleway",
                                                    "size": 10,
                                                },
                                                xaxis={
                                                    "autorange": True,
                                                    "range": [
                                                        1,
                                                        23,
                                                    ],
                                                    "rangeselector": {
                                                        "buttons": [
                                                            {
                                                                "count": 1,
                                                                "step": "year",
                                                                "label": "1Y",
                                                                "stepmode": "backward",
                                                            },
                                                            {
                                                                "count": 2,
                                                                "step": "year",
                                                                "label": "2Y",
                                                                "stepmode": "backward",
                                                            },
                                                            {
                                                                "count": 3,
                                                                "step": "year",
                                                                "label": "3Y",
                                                                "stepmode": "backward",
                                                            },
                                                            {
                                                                "label": "All",
                                                                "step": "all",
                                                            },
                                                        ]
                                                    },
                                                    "showline": True,
                                                    "type": "date",
                                                    "zeroline": False,
                                                },
                                                yaxis={
                                                    "autorange": True,
                                                    "showline": True,
                                                    "title": "$",
                                                    "type": "linear",
                                                    "zeroline": False,
                                                },
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 5
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["RL Improvement"],
                                        className="subtitle padded",
                                    ),
                                    html.Table(
                                        make_dash_table(df_test2)
                                    ),
                                ],
                                className=" ten columns",
                            )
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
