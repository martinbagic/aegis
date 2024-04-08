from dash import callback, Output, Input, ctx, ALL, State

import logging
from aegis.utilities.container import Container
from visor.utilities import default_selection_states, log_debug, get_base_dir
from visor.tab_plot import prep_fig
from visor.tab_plot.prep_setup import FIG_SETUP


def gen_fig(fig_name, selected_sims, containers):
    """Generates a figure using the figure setup"""

    # Extract setup
    fig_setup = FIG_SETUP[fig_name]

    # Prepare x and y data
    prep_x = fig_setup["prep_x"]
    prep_y = fig_setup["prep_y"]
    ys = [prep_y(containers[sim]) for sim in selected_sims]
    xs = [prep_x(containers[sim], y=y) for sim, y in zip(selected_sims, ys)]

    # Generate go figure
    prep_figure = getattr(prep_fig, fig_setup["prep_figure"])
    figure = prep_figure(fig_name, xs, ys, selected_sims)

    return figure


@callback(
    Output("plot-view-button", "className"),
    Input({"type": "selection-state", "index": ALL}, "data"),
    State("plot-view-button", "className"),
)
@log_debug
def disable_plot_tab(data, className):
    className = className.replace(" disabled", "")
    if data == [] or all(not selected for filename, selected in data):
        return className + " disabled"
    return className


@callback(
    [Output(key, "figure") for key in FIG_SETUP.keys()],
    Input("plot-view-button", "n_clicks"),
    Input("reload-plots-button", "n_clicks"),
    State({"type": "selection-state", "index": ALL}, "data"),
)
@log_debug
def update_plot_tab(n_clicks1, n_clicks2, selection_states):
    """
    Update plots whenever someone clicks on the plot button or the reload button.
    """

    # If initial call, run the function so that the figures get initialized
    if ctx.triggered_id is None:  # if initial call
        selection_states = list(default_selection_states)

    containers = {}
    base_dir = get_base_dir()
    for filename, selected in selection_states:
        if selected and filename not in containers:
            results_path = base_dir / filename
            logging.info(f"Fetching data from {results_path}")
            containers[filename] = Container(base_dir / filename)

    selected_sims = [filename for filename, selected in selection_states if selected]
    logging.info("Plotting: " + ", ".join(selected_sims))

    # Prepare figures
    # BUG no data saved yet on running simulations or interrupted simulations
    return [gen_fig(fig_name, selected_sims, containers) for fig_name in FIG_SETUP]
