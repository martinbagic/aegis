from dash import callback, Output, Input, ctx

from aegis.help.container import Container
from aegis.visor import funcs
from aegis.visor.tab_plot import prep_fig
from aegis.visor.tab_plot.prep_setup import FIG_SETUP
from aegis.visor.tab_list.callbacks_list import SELECTION


containers = {}


def gen_fig(fig_name):
    """Generates a figure using the figure setup"""

    # Extract setup
    fig_setup = FIG_SETUP[fig_name]

    # Prepare x and y data
    prep_x = fig_setup["prep_x"]
    prep_y = fig_setup["prep_y"]
    ys = [prep_y(containers[sim]) for sim in SELECTION]
    xs = [prep_x(containers[sim], y=y) for sim, y in zip(SELECTION, ys)]

    # Generate go figure
    prep_figure = getattr(prep_fig, fig_setup["prep_figure"])
    figure = prep_figure(fig_name, xs, ys)

    return figure


@callback(
    [Output(key, "figure") for key in FIG_SETUP.keys()],
    Input("plot-view-button", "n_clicks"),
    Input("reload-plots-button", "n_clicks"),
    prevent_initial_call=True,
)
@funcs.print_function_name
def update_plot_tab(*_):
    """
    Update plots whenever someone clicks on the plot button or the reload button.
    """
    global containers

    # Clear out containers if the user clicked the reload button
    triggered = ctx.triggered_id
    if triggered == "reload-plots-button":
        containers = {}

    # Load selected containers
    for sim in SELECTION:
        if sim not in containers:
            containers[sim] = Container(funcs.BASE_DIR / sim)

    # Prepare figures
    # BUG no data saved yet on running simulations or interrupted simulations
    return [gen_fig(fig_name) for fig_name in FIG_SETUP]
