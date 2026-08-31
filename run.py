"""Development entry point.

Debug mode is off unless GRAPHNEXUS_DEBUG is set, and the recursion limit is
left alone: every traversal in graphnexus.algorithms is iterative, so raising
it is unnecessary. The original set it to 10,000,000, which does not grow the
C stack and so turns deep recursion into a segfault rather than an exception.

For anything beyond local development, serve the app with a WSGI server:

    gunicorn "graphnexus:create_app()"
"""

import os

from graphnexus import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.environ.get("GRAPHNEXUS_HOST", "127.0.0.1"),
        port=int(os.environ.get("GRAPHNEXUS_PORT", 5000)),
        debug=os.environ.get("GRAPHNEXUS_DEBUG") == "1",
    )
