import dash
from dash import html, dcc
from aegis.visor.pages.tab_plot.prep_setup import FIG_SETUP, needs_slider
from aegis.visor.utilities import log_funcs, utilities
from aegis.utilities.container import Container
from aegis.visor.pages.tab_plot.tabs import get_tabs_layout

dash.register_page(__name__, path="/plot", name="plot")


def get_preface():
    simnames = [Container(path).basepath.stem for path in utilities.get_sim_paths()]
    dropdown_options = [simname for simname in simnames]
    # TODO plot initially
    if dropdown_options:
        if "default" in dropdown_options:
            dropdown_option_chosen = "default"
        else:
            dropdown_option_chosen = dropdown_options[0]
    else:
        dropdown_option_chosen = None
    dropdown_option_chosen = None
    preface = [
        html.Div(
            children=[
                # TODO change text
                html.P(
                    [
                        """
                        This is the plot tab. Here you can explore the simulation visually.
                        You can also download the figures and the data used for plotting.
                        The figures are interactive – if multiple simulations are displayed, you can click on
                        simulation IDs to toggle their visibility; you can also zoom in and out.
                        For figures that show time-specific data, you can use sliders to change the time point plotted.
                        """,
                    ],
                    style={"margin-bottom": "2rem"},
                ),
                html.Button(
                    "reload",
                    "reload-plots-button",
                    className="control-element",
                ),
                dcc.Dropdown(id="dropdown", options=dropdown_options, value=dropdown_option_chosen, multi=True),
                get_tabs_layout(),
            ],
        )
    ]
    return preface


@log_funcs.log_debug
def get_plot_layout():
    return html.Div(
        id="plot-section",
        # style={"display": "none"},
        children=get_preface(),
        # + [
        #     html.Div(
        #         id="figure-container",
        #         children=[
        #             html.Div(
        #                 [
        #                     # html.Div(
        #                     #     [
        #                     #         dcc.Graph(
        #                     #             # id=figure_id,
        #                     #             id={"type": "graph-figure", "index": figure_id},
        #                     #             config={"displayModeBar": False},
        #                     #             className="figure",
        #                     #         ),
        #                     #         (
        #                     #             dcc.Slider(
        #                     #                 min=0,
        #                     #                 max=1,
        #                     #                 step=1,
        #                     #                 value=0,
        #                     #                 marks=None,
        #                     #                 tooltip={},
        #                     #                 id={"type": "graph-slider", "index": figure_id},
        #                     #             )
        #                     #             if needs_slider(figure_id)
        #                     #             else None
        #                     #         ),
        #                     #     ],
        #                     #     style={"padding-right": "0.9rem", "margin-left": "0.6rem"},
        #                     # ),
        #                     # html.Div(
        #                     #     children=[
        #                     #         html.P(
        #                     #             children=info["title"],
        #                     #             className="figure-title",
        #                     #         ),
        #                     #         html.P(
        #                     #             children=info["description"],
        #                     #             className="figure-description",
        #                     #         ),
        #                     #         html.Button(
        #                     #             "download figure",
        #                     #             id={"type": "figure-download-button", "index": figure_id},
        #                     #         ),
        #                     #         dcc.Download(id={"type": "figure-dcc-download", "index": figure_id}),
        #                     #         # dcc.Slider(0,100)
        #                     #     ]
        #                     # ),
        #                 ],
        #                 className="figure-card",
        #             )
        #             for figure_id, info in FIG_SETUP.items()
        #         ],
        #     ),
        # ],
        # + [
        #     dcc.Graph(id="figurex"),
        #     html.Div(
        #         [
        #             dcc.Slider(
        #                 id="slider",
        #                 min=1,
        #                 max=10,
        #                 step=1,
        #                 value=5,
        #                 updatemode="drag",
        #             ),
        #         ],
        #         # style={"width": "400px"},
        #     ),
        # ],
    )


layout = get_plot_layout()
# layout = PREFACE

# layout = [
#     html.Div(
#         # id="landing-section",
#         children=[
#             html.P("asdfe"),
#         ],
#     )
# ]
