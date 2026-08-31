"""GraphNexus, a block-paged graph store with a web front end.

The original module created the Flask app at import time and imported its own
routes from inside ``__init__``, which makes the package impossible to import
for testing without also standing up the web layer. This uses an application
factory instead.
"""

from __future__ import annotations

import os

from flask import Flask
from flask_wtf.csrf import CSRFProtect

__version__ = "1.0.0"


def create_app(config_object=None) -> Flask:
    from config import Config

    app = Flask(__name__)
    app.config.from_object(config_object or Config)
    os.makedirs(app.config["DATA_DIR"], exist_ok=True)

    # Registers the csrf_token() template global and protects every POST
    # route, including the ones that are not driven by a WTForms form.
    CSRFProtect(app)

    from .routes import bp

    app.register_blueprint(bp)
    return app
