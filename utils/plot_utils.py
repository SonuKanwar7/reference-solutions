from dataclasses import dataclass
from typing import Callable, List, Optional, Union, Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt

# --- Data classes ---
@dataclass
class PiecewiseLine:
    y_func: Callable[[np.ndarray], np.ndarray]
    cond_func: Callable[[np.ndarray], np.ndarray] = lambda x: x == x
    label: str = ""
    highlight_x: Optional[List[float]] = None
    color: Optional[str] = None

@dataclass
class VerticalLine:
    x: float
    label: str = ""
    color: Optional[str] = None


def HorizontalLine(
    y: float,
    cond_func: Callable[[np.ndarray], np.ndarray],
    label: str = "",
    highlight_x: Optional[List[float]] = None,
    color: Optional[str] = None,
) -> PiecewiseLine:
    return PiecewiseLine(
        y_func=lambda x: np.full_like(x, y),
        cond_func=cond_func,
        label=label,
        highlight_x=highlight_x,
        color=color,
    )


PlotItem = Union[PiecewiseLine, VerticalLine]

# --- Subplot configuration class ---
@dataclass
class SubplotConfig:
    """Configuration for a single subplot"""
    lines: Sequence[PlotItem]
    x_min: float = -10
    x_max: float = 10
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    title: str = ""
    xlabel: str = "x"
    ylabel: str = "y"

# --- Core plotting function ---
def _plot_on_axes(
    ax: plt.Axes,
    config: SubplotConfig,
    axis_color: str = "black",
    axis_width: float = 1.5,
):
    """Plot on given axes using SubplotConfig"""
    x = np.linspace(config.x_min, config.x_max, 1000)
    
    for item in config.lines:
        c = item.color
        if isinstance(item, PiecewiseLine):
            mask = item.cond_func(x)
            y = np.full_like(x, np.nan, dtype=float)
            y[mask] = item.y_func(x[mask])
            ax.plot(x, y, label=item.label, color=c)

            if item.highlight_x:
                for xp in item.highlight_x:
                    xp_arr = np.array([xp])
                    if item.cond_func(xp_arr)[0]:
                        yp = item.y_func(xp_arr)[0]
                        ax.plot(xp, yp, "o", color=c)
                        ax.annotate(
                            f"({xp}, {yp:.2f})",
                            (xp, yp),
                            textcoords="offset points",
                            xytext=(5, 5),
                            fontsize=8,
                            color=c,
                        )

        elif isinstance(item, VerticalLine):
            ymin_, ymax_ = (config.y_min, config.y_max) if (
                config.y_min is not None and config.y_max is not None
            ) else ax.get_ylim()
            ax.vlines(item.x, ymin_, ymax_, label=item.label, color=c)

    ax.set_xlim(config.x_min, config.x_max)
    if config.y_min is not None and config.y_max is not None:
        ax.set_ylim(config.y_min, config.y_max)

    # Dotted axes
    ax.axhline(0, color=axis_color, linewidth=axis_width, linestyle=":")
    ax.axvline(0, color=axis_color, linewidth=axis_width, linestyle=":")

    ax.grid(True)
    if any(item.label for item in config.lines):
        ax.legend()
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title)

# --- Public plotting function ---
def plot_lines(
    lines: Sequence[PlotItem],
    x_min: float = -10,
    x_max: float = 10,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    title: str = "Piecewise Line Plot",
    xlabel: str = "x",
    ylabel: str = "y",
    axis_color: str = "black",
    axis_width: float = 1.5,
) -> Tuple[plt.Figure, plt.Axes]:
    """Public function for single plot"""
    config = SubplotConfig(
        lines=lines,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )
    
    fig, ax = plt.subplots()
    _plot_on_axes(ax, config, axis_color, axis_width)
    return fig, ax

# --- Combined subplot function ---
def plot_combined(
    subplot_configs: Sequence[SubplotConfig],
    nrows: int,
    ncols: int,
    figsize: Tuple[float, float] = (12, 8),
    main_title: Optional[str] = None,
    axis_color: str = "black",
    axis_width: float = 1.5,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create combined subplots from multiple SubplotConfig objects
    
    Args:
        subplot_configs: Sequence of SubplotConfig objects
        nrows: Number of rows in subplot grid
        ncols: Number of columns in subplot grid
        figsize: Overall figure size
        main_title: Optional title for entire figure
        axis_color: Color for dotted axes
        axis_width: Width for dotted axes
    
    Returns:
        (fig, axes) tuple
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    # Plot each subplot configuration
    for i, config in enumerate(subplot_configs):
        if i < len(axes_flat):
            _plot_on_axes(
                axes_flat[i], 
                config, 
                axis_color, 
                axis_width
            )
    
    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
    
    if main_title:
        fig.suptitle(main_title, fontsize=16)
    
    fig.tight_layout()
    if main_title:
        fig.subplots_adjust(top=0.92)
        
    return fig, axes