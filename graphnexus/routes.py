"""HTTP routes.

The active graph is held in the signed session cookie, not in a module-level
global. The original stored it in a module global, so with Flask's threaded
development server two people using the app at the same time overwrote each
other's selection.
"""

from __future__ import annotations

import os

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from .forms import (
    NeighborsForm,
    NodeForm,
    NodePairForm,
    RankRangeForm,
    ScoreRangeForm,
    UploadForm,
)
from .graphstore import (
    SOURCE_FILENAME,
    GraphError,
    GraphStore,
    build_graph,
    delete_graph,
    list_graphs,
    sanitize_graph_name,
)

bp = Blueprint("main", __name__)

SESSION_KEY = "graph_name"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def data_dir() -> str:
    return current_app.config["DATA_DIR"]


def open_active_graph() -> GraphStore | None:
    """Open the graph named in the session, or None if there isn't a valid one."""
    name = session.get(SESSION_KEY)
    if not name:
        return None
    try:
        safe = sanitize_graph_name(name)
    except GraphError:
        session.pop(SESSION_KEY, None)
        return None
    try:
        return GraphStore(
            os.path.join(data_dir(), safe),
            buffer_blocks=current_app.config["BUFFER_POOL_BLOCKS"],
        )
    except GraphError:
        session.pop(SESSION_KEY, None)
        return None


@bp.app_context_processor
def inject_nav_state():
    """Give every template the active graph name and the list of built graphs."""
    return {
        "active_graph": session.get(SESSION_KEY),
        "available_graphs": list_graphs(data_dir()),
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@bp.route("/")
def home():
    return render_template("home.html")


@bp.route("/graphs", methods=["GET", "POST"])
def graphs_page():
    form = UploadForm()

    if form.validate_on_submit():
        upload = form.graph_file.data
        filename = secure_filename(upload.filename or "")
        extension = os.path.splitext(filename)[1].lower()
        if extension not in current_app.config["ALLOWED_EXTENSIONS"]:
            flash(f"{extension or 'That file type'} is not supported.", "error")
            return redirect(url_for("main.graphs_page"))

        try:
            name = sanitize_graph_name(filename)
        except GraphError as exc:
            flash(str(exc), "error")
            return redirect(url_for("main.graphs_page"))

        graph_dir = os.path.join(data_dir(), name)
        os.makedirs(graph_dir, exist_ok=True)
        source_path = os.path.join(graph_dir, SOURCE_FILENAME)
        upload.save(source_path)

        weighted = form.graph_type.data == "weighted"
        try:
            meta = build_graph(source_path, graph_dir, weighted=weighted)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            delete_graph(data_dir(), name)
            flash(f"Could not build {name}: {exc}", "error")
            return redirect(url_for("main.graphs_page"))

        session[SESSION_KEY] = name
        flash(
            f"Built {name}: {meta.nodes:,} nodes, {meta.edges:,} edges "
            f"in {meta.build_ms:,} ms.",
            "success",
        )
        for warning in meta.warnings:
            flash(warning, "warning")
        return redirect(url_for("main.metadata_page"))

    return render_template("graphs.html", form=form)


@bp.route("/graphs/<name>/select", methods=["POST"])
def select_graph(name: str):
    try:
        safe = sanitize_graph_name(name)
    except GraphError as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.graphs_page"))
    if safe not in list_graphs(data_dir()):
        flash(f"No graph named {safe}.", "error")
        return redirect(url_for("main.graphs_page"))
    session[SESSION_KEY] = safe
    flash(f"{safe} is now the active graph.", "success")
    return redirect(url_for("main.metadata_page"))


@bp.route("/graphs/<name>/delete", methods=["POST"])
def remove_graph(name: str):
    try:
        safe = sanitize_graph_name(name)
    except GraphError as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.graphs_page"))
    delete_graph(data_dir(), safe)
    if session.get(SESSION_KEY) == safe:
        session.pop(SESSION_KEY, None)
    flash(f"Deleted {safe}.", "success")
    return redirect(url_for("main.graphs_page"))


@bp.route("/metadata")
def metadata_page():
    store = open_active_graph()
    if store is None:
        flash("Load a graph first.", "warning")
        return redirect(url_for("main.graphs_page"))
    with store:
        return render_template("metadata.html", meta=store.meta)


# ---------------------------------------------------------------------------
# Query pages
# ---------------------------------------------------------------------------


def _single_node_query(title, description, method_name, result_template):
    store = open_active_graph()
    if store is None:
        flash("Load a graph first.", "warning")
        return redirect(url_for("main.graphs_page"))

    with store:
        form = NodeForm()
        form.apply_bounds(store.meta.nodes - 1)
        result = None
        if form.validate_on_submit():
            try:
                result = getattr(store, method_name)(form.node.data)
            except GraphError as exc:
                flash(str(exc), "error")
        return render_template(
            "query.html",
            title=title,
            description=description,
            form=form,
            result=result,
            result_template=result_template,
            meta=store.meta,
        )


@bp.route("/indegree", methods=["GET", "POST"])
def indegree_page():
    return _single_node_query(
        "In-degree",
        "How many edges point at a node.",
        "in_degree",
        "partials/scalar.html",
    )


@bp.route("/outdegree", methods=["GET", "POST"])
def outdegree_page():
    return _single_node_query(
        "Out-degree",
        "How many edges leave a node.",
        "out_degree",
        "partials/scalar.html",
    )


@bp.route("/pagerank", methods=["GET", "POST"])
def rank_page():
    return _single_node_query(
        "PageRank",
        "A node's PageRank score and its position in the overall ranking.",
        "pagerank_of",
        "partials/pagerank.html",
    )


@bp.route("/shortest-distance", methods=["GET", "POST"])
def shortest_distance_page():
    store = open_active_graph()
    if store is None:
        flash("Load a graph first.", "warning")
        return redirect(url_for("main.graphs_page"))

    with store:
        form = NodePairForm()
        form.apply_bounds(store.meta.nodes - 1)
        result = None
        if form.validate_on_submit():
            try:
                result = store.shortest_distance(form.source.data, form.target.data)
            except GraphError as exc:
                flash(str(exc), "error")
        return render_template(
            "query.html",
            title="Shortest distance",
            description=(
                "Fewest edges between two nodes. Weighted graphs use Dijkstra, "
                "unweighted use breadth-first search."
            ),
            form=form,
            result=result,
            result_template="partials/distance.html",
            meta=store.meta,
        )


@bp.route("/components", methods=["GET", "POST"])
def component_page():
    store = open_active_graph()
    if store is None:
        flash("Load a graph first.", "warning")
        return redirect(url_for("main.graphs_page"))

    with store:
        form = NodePairForm()
        form.apply_bounds(store.meta.nodes - 1)
        result = None
        if form.validate_on_submit():
            try:
                result = store.same_components(form.source.data, form.target.data)
            except GraphError as exc:
                flash(str(exc), "error")
        return render_template(
            "query.html",
            title="Shared components",
            description="Whether two nodes sit in the same strongly or weakly connected component.",
            form=form,
            result=result,
            result_template="partials/components.html",
            meta=store.meta,
        )


@bp.route("/edge", methods=["GET", "POST"])
def edge_page():
    store = open_active_graph()
    if store is None:
        flash("Load a graph first.", "warning")
        return redirect(url_for("main.graphs_page"))

    with store:
        form = NodePairForm()
        form.apply_bounds(store.meta.nodes - 1)
        result = None
        if form.validate_on_submit():
            try:
                result = store.has_edge(form.source.data, form.target.data)
            except GraphError as exc:
                flash(str(exc), "error")
        return render_template(
            "query.html",
            title="Edge lookup",
            description=(
                "Whether one node links to another, resolved through the static "
                "hash index rather than by scanning the adjacency run."
            ),
            form=form,
            result=result,
            result_template="partials/edge.html",
            meta=store.meta,
        )


@bp.route("/score-range", methods=["GET", "POST"])
def score_range_page():
    store = open_active_graph()
    if store is None:
        flash("Load a graph first.", "warning")
        return redirect(url_for("main.graphs_page"))

    with store:
        form = ScoreRangeForm()
        result = None
        if form.validate_on_submit():
            try:
                result = store.nodes_by_score(
                    form.low.data, form.high.data, form.limit.data
                )
            except GraphError as exc:
                flash(str(exc), "error")
        return render_template(
            "query.html",
            title="Score range",
            description=(
                "Every node whose PageRank falls inside a range, found by "
                "descending the B+-tree and walking its linked leaves."
            ),
            form=form,
            result=result,
            result_template="partials/scorerange.html",
            meta=store.meta,
        )


@bp.route("/knn", methods=["GET", "POST"])
def knn_page():
    store = open_active_graph()
    if store is None:
        flash("Load a graph first.", "warning")
        return redirect(url_for("main.graphs_page"))

    with store:
        form = NeighborsForm()
        form.apply_bounds(store.meta.nodes - 1)
        result = None
        if form.validate_on_submit():
            try:
                result = store.knn(form.node.data, form.k.data)
            except GraphError as exc:
                flash(str(exc), "error")
        return render_template(
            "query.html",
            title="Nearest neighbours",
            description="The k closest reachable nodes, ordered by distance.",
            form=form,
            result=result,
            result_template="partials/knn.html",
            meta=store.meta,
        )


@bp.route("/rank-list", methods=["GET", "POST"])
def rank_list_page():
    store = open_active_graph()
    if store is None:
        flash("Load a graph first.", "warning")
        return redirect(url_for("main.graphs_page"))

    with store:
        form = RankRangeForm()
        result = None
        if form.validate_on_submit():
            try:
                result = store.rank_list(form.first.data, form.last.data)
            except GraphError as exc:
                flash(str(exc), "error")
        return render_template(
            "query.html",
            title="Rank range",
            description=(
                "Nodes between two PageRank positions. Ranks are stored as a "
                "sorted array, so this is a sequential block scan."
            ),
            form=form,
            result=result,
            result_template="partials/ranklist.html",
            meta=store.meta,
        )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@bp.app_errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404,
                           message="That page does not exist."), 404


@bp.app_errorhandler(413)
def too_large(_):
    limit = current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return render_template("error.html", code=413,
                           message=f"That file is larger than the {limit} MB limit."), 413


@bp.app_errorhandler(500)
def server_error(_):
    return render_template("error.html", code=500,
                           message="Something went wrong handling that request."), 500
