"""Form definitions.

Node ids are validated against the loaded graph's real bounds, so an
out-of-range id is rejected at the form layer instead of reaching storage. The
original imported BeautifulSoup here and never used it.
"""

from __future__ import annotations

from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import DecimalField, IntegerField, SelectField, SubmitField
from wtforms.validators import InputRequired, NumberRange


class UploadForm(FlaskForm):
    graph_file = FileField(
        "Edge list",
        validators=[
            FileRequired(message="Choose an edge list to upload."),
            FileAllowed(
                ["txt", "csv", "tsv", "edges"],
                message="Upload a .txt, .csv, .tsv or .edges file.",
            ),
        ],
    )
    graph_type = SelectField(
        "Edge weights",
        choices=[("unweighted", "Unweighted"), ("weighted", "Weighted")],
        default="unweighted",
    )
    submit = SubmitField("Build graph")


class NodeForm(FlaskForm):
    """Single node id, bounded by the active graph."""

    node = IntegerField(
        "Node",
        validators=[InputRequired(message="Enter a node id.")],
    )
    submit = SubmitField("Run query")

    def apply_bounds(self, max_node: int) -> None:
        self.node.validators = [
            InputRequired(message="Enter a node id."),
            NumberRange(0, max_node, message=f"Node ids run from 0 to {max_node}."),
        ]


class NodePairForm(FlaskForm):
    """Two node ids, both bounded by the active graph."""

    source = IntegerField(
        "From node", validators=[InputRequired(message="Enter a source node.")]
    )
    target = IntegerField(
        "To node", validators=[InputRequired(message="Enter a target node.")]
    )
    submit = SubmitField("Run query")

    def apply_bounds(self, max_node: int) -> None:
        message = f"Node ids run from 0 to {max_node}."
        for field in (self.source, self.target):
            field.validators = [
                InputRequired(message="Enter a node id."),
                NumberRange(0, max_node, message=message),
            ]


class NeighborsForm(FlaskForm):
    """Node plus a neighbour count."""

    node = IntegerField(
        "Node", validators=[InputRequired(message="Enter a node id.")]
    )
    k = IntegerField(
        "Neighbours to return",
        validators=[
            InputRequired(message="Enter how many neighbours to return."),
            NumberRange(1, 1000, message="Ask for between 1 and 1000 neighbours."),
        ],
    )
    submit = SubmitField("Run query")

    def apply_bounds(self, max_node: int) -> None:
        self.node.validators = [
            InputRequired(message="Enter a node id."),
            NumberRange(0, max_node, message=f"Node ids run from 0 to {max_node}."),
        ]


class RankRangeForm(FlaskForm):
    """An inclusive rank range."""

    first = IntegerField(
        "From rank",
        validators=[
            InputRequired(message="Enter the first rank."),
            NumberRange(1, message="Ranks start at 1."),
        ],
    )
    last = IntegerField(
        "To rank",
        validators=[
            InputRequired(message="Enter the last rank."),
            NumberRange(1, message="Ranks start at 1."),
        ],
    )
    submit = SubmitField("Run query")


class ScoreRangeForm(FlaskForm):
    """An inclusive PageRank score range, answered by the B+-tree."""

    low = DecimalField(
        "Lowest score",
        places=None,
        validators=[
            InputRequired(message="Enter the lower bound."),
            NumberRange(0, 1, message="PageRank scores lie between 0 and 1."),
        ],
    )
    high = DecimalField(
        "Highest score",
        places=None,
        validators=[
            InputRequired(message="Enter the upper bound."),
            NumberRange(0, 1, message="PageRank scores lie between 0 and 1."),
        ],
    )
    limit = IntegerField(
        "Maximum rows",
        default=100,
        validators=[
            InputRequired(message="Enter a row limit."),
            NumberRange(1, 10_000, message="Ask for between 1 and 10,000 rows."),
        ],
    )
    submit = SubmitField("Run query")
