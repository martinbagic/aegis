import dash
from dash import html, dcc, dash_table
from aegis.visor.utilities import log_funcs

from aegis.visor.utilities.utilities import OUTPUT_SPECIFICATIONS


dash.register_page(__name__, path="/simlog", name="simlog")


@log_funcs.log_debug
def make_output_specification_table():
    """

    Documentation for plotly datatables: https://dash.plotly.com/datatable
    """
    data = [
        {key: specs.get(key, "!!! nan") for key in OUTPUT_SPECIFICATIONS[0].keys()} for specs in OUTPUT_SPECIFICATIONS
    ]
    columns = [{"id": c, "name": c} for c in OUTPUT_SPECIFICATIONS[0].keys()]
    return dash_table.DataTable(
        data=data,
        columns=columns,
        style_data={"whiteSpace": "normal", "height": "auto"},
    )


PREFACE = [
    html.Div(
        children=[
            # TODO change text
            html.P(
                [
                    """
                    This is the list tab. Here you can see which simulations you have started and what their status is, including their metadata.
                    You can also delete simulations and mark which simulations you would like to display in the plot tab.

                    Note that the 'default' simulation cannot be deleted – it comes with the installation and can be used
                    as an exploratory example. All parameters for the default simulation have default values.
                    """,
                ],
                style={"margin-bottom": "2rem"},
            ),
            # make_output_specification_table(),
        ],
    )
]


@log_funcs.log_debug
def get_list_layout():
    # selection_states = [
    #     dcc.Store({"type": "selection-state", "index": sim}, data=False)
    #     for sim in funcs.get_sims()
    # ]
    return html.Div(
        id="simlog-section",
        # style={"display": "none"},
        children=PREFACE
        + [
            # html.Div(id="selection-states", children=selection_states),
            html.Div(id="simlog-section-table", children=[]),
        ],
    )


layout = get_list_layout()
