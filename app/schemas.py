"""Pydantic models shared by the LLM (structured output) and the API."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

PLOT_KINDS = ("hist", "box", "scatter", "bar", "count", "heatmap_corr")

# What each plot kind needs - shown to the LLM and enforced by plots.py
PLOT_RULES = """\
- hist:          x = numeric column (y unused)
- box:           y = numeric column, x = optional categorical column
- scatter:       x = numeric column, y = numeric column
- bar:           x = categorical column, y = numeric column (mean of y per x)
- count:         x = categorical column (y unused)
- heatmap_corr:  no columns needed (correlation of all numeric columns); at most ONE"""


class PlotSpec(BaseModel):
    kind: Literal["hist", "box", "scatter", "bar", "count", "heatmap_corr"]
    x: Optional[str] = Field(None, description="Column for the x axis, if needed")
    y: Optional[str] = Field(None, description="Column for the y axis, if needed")
    reason: str = Field(description="One short sentence: why this plot is useful")


class DataUnderstanding(BaseModel):
    summary: str = Field(description="2-4 sentence description of what the dataset appears to be")
    suggested_target: str = Field(description="Most likely target column to predict")
    task_type: Literal["classification", "regression"]
    drop_columns: list[str] = Field(
        default_factory=list,
        description="Columns to exclude from training: IDs, free text, constants, leaky columns",
    )
    warnings: list[str] = Field(default_factory=list)
    plots: list[PlotSpec] = Field(default_factory=list)
