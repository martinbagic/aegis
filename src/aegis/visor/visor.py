"""
Import callbacks. They modify Output's when Input's are triggered. Callbacks add reactivity to the app.
"""

from dash import Dash

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    update_title="",
    # *.css in assets are automatically imported; they need to be explicitly ignored
    assets_ignore="styles-dark.css",
    use_pages=True,
)


def run(environment):

    from aegis.visor.app.layout import app_layout
    import aegis.visor.app.callbacks
    import aegis.visor.pages.tab_config.callbacks_config
    import aegis.visor.pages.tab_plot.callbacks_plot
    import aegis.visor.pages.tab_plot.callbacks_download
    import aegis.visor.pages.tab_simlog.callbacks_list

    from .config import get_config

    config = get_config(environment=environment)

    app._favicon = "favicon.ico"
    app.title = "AEGIS visualizer"
    app.layout = app_layout
    app.run(debug=config.debug)
