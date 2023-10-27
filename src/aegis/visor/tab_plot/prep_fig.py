from aegis.visor.tab_plot.setup import FIG_SETUP
from aegis.visor.tab_list.callbacks_list import SELECTION
import plotly.graph_objs as go

FIG_LAYOUT = dict(
    width=300,
    height=300,
    margin={"t": 0, "r": 0, "b": 0, "l": 0},
    plot_bgcolor="rgba(190, 189, 183, 0.0)",
    paper_bgcolor="rgba(24, 25, 27, 0)",
    font_color="white",
    showlegend=False,
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        nticks=10,
        gridcolor="rgb(46, 49, 51)",
        linecolor="rgb(46, 49, 51)",
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        nticks=10,
        gridcolor="rgb(46, 49, 51)",
        linecolor="rgb(46, 49, 51)",
    ),
)


def make_scatter_figure(id_, xs, ys):
    figure = go.Figure(
        data=[
            go.Scatter(x=x, y=y, mode="markers", name=sim)
            for x, y, sim in zip(xs, ys, SELECTION)
        ],
        layout=go.Layout({**FIG_LAYOUT, **FIG_SETUP[id_]["figure_layout"]}),
    )

    # Compute maximum axes limits
    maxx = max(max(x) for x in xs) if ys else 1
    maxy = max(max(y) for y in ys) if ys else 1

    # Update axes
    figure.update_xaxes(
        range=[0, 1 if maxx < 1 else maxx * 1.05],
    )
    figure.update_yaxes(
        range=[0, 1.05 if maxy < 1 else maxy * 1.1],
    )

    # Custom plots
    if id_ == "birth structure":
        figure.update_yaxes(range=[0, maxy * 1.05])

    return figure


def make_hist_figure(id_, xs, ys):
    figure = go.Figure(
        data=[go.Histogram(x=y, name=sim) for y, sim in zip(ys, SELECTION)],
        layout=go.Layout({**FIG_LAYOUT, **FIG_SETUP[id_]["figure_layout"]}),
    )

    return figure
