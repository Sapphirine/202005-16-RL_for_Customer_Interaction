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
df_para = pd.read_csv(DATA_PATH.joinpath("df_parameters2.csv"))
df_it = pd.read_csv(DATA_PATH.joinpath("df_it2.csv"))
df_im = pd.read_csv(DATA_PATH.joinpath("df_im2.csv"))

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
                                    html.H5("Indirect RL"),
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
                                    html.H6(
                                        ["Model Parameters"],
                                        className="subtitle padded",
                                    ),
                                    html.Table(
                                        make_dash_table(df_para)
                                    ),
                                ],
                                className=" ten columns",
                            )
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
                                                )
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
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("Performance over iterations", className="subtitle padded"),
                                    dcc.Graph(
                                        id="graph-5",
                                        figure={
                                            "data": [
                                                go.Scatter(
                                                    x=df_it["Iteration"],
                                                    y=df_it["Baseline"],
                                                    line={"color": "#97151c"},
                                                    mode="lines",
                                                    name="Baseline",
                                                ),
                                                go.Scatter(
                                                    x=df_it["Iteration"],
                                                    y=df_it[
                                                        "Direct RL"
                                                    ],
                                                    line={"color": "#b5b5b5"},
                                                    mode="lines",
                                                    name="Direct RL",
                                                )
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
                                                        20,
                                                    ],
                                                    "showline": True,
                                                    "type": "linear",
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
                                        make_dash_table(df_im)
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
