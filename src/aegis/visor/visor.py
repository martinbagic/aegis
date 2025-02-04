"""
Import callbacks. They modify Output's when Input's are triggered. Callbacks add reactivity to the app.
"""

from dash import Dash


def run(environment):
    from . import config

    config.set(environment=environment)

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        update_title="",
        # *.css in assets are automatically imported; they need to be explicitly ignored
        assets_ignore="styles-dark.css",
        use_pages=True,
    )

    from aegis.visor.app.layout import app_layout

    app._favicon = "favicon.ico"
    app.title = "AEGIS visualizer"
    app.layout = app_layout
    app.run(debug=config.config.debug_mode)
