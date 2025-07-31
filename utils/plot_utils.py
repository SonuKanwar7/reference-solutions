from __future__ import annotations
"""Payoff plotting utilities.

Goals:
- Single, simple API that subsumes earlier `PiecewiseLine` / `HorizontalLine` usage.
- Support for: piecewise masking, vertical lines, point highlights, subplots.

Core Concepts
-------------
RangeVar:
    Named domain (1D grid). Behaves like a NumPy array in expressions.
Payoff:
    Labeled data series produced from either an ndarray or a callable over the domain.
    Optional mask => piecewise region (outside rendered as NaN gaps).
    Optional highlight_x list marks specific x values (if inside mask).
VerticalLine:
    Simple structural marker drawn over each subplot.

Primary Functions
-----------------
plot_payoffs(payoffs, ...):
    Plot a single group of payoffs (optionally with vertical lines) sharing one domain.
plot_panels(panels, ...):
    Create subplots (each panel has its own domain/payoffs/settings).

Future Work
-----------
- Build utility wrappers later (e.g. for European call: Payoff('Call', call(S,K), ...)).
"""
from dataclasses import dataclass
from typing import Callable, Sequence, Optional, Union, List, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt

ArrayFunc = Callable[[np.ndarray], np.ndarray]
MaskType = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]

# -----------------------------------------------------------------------------
# Range variable
# -----------------------------------------------------------------------------
@dataclass
class RangeVar:
    name: str
    values: np.ndarray  # 1D
    __array_priority__ = 1000

    def __array__(self):  # allow numpy ops directly
        return self.values

    @classmethod
    def linspace(cls, name: str, start: float, stop: float, num: int) -> RangeVar:
        return cls(name, np.linspace(start, stop, num))

    @classmethod
    def arange(cls, name: str, start: float, stop: float, step: float) -> RangeVar:
        return cls(name, np.arange(start, stop + 1e-12, step))

# -----------------------------------------------------------------------------
# Payoff (with optional mask, highlight, color)
# -----------------------------------------------------------------------------
@dataclass
class Payoff:
    label: str
    data: Union[np.ndarray, ArrayFunc]
    domain: RangeVar
    mask: Optional[MaskType] = None
    highlight_x: Optional[Sequence[float]] = None
    color: Optional[str] = None

    def evaluate(self) -> np.ndarray:
        x = self.domain.values
        y = self.data(x) if callable(self.data) else self.data
        y = np.asarray(y, dtype=float)
        if y.shape != x.shape:
            raise ValueError(f"Payoff '{self.label}' shape {y.shape} does not match domain {x.shape}")
        if self.mask is not None:
            m = self.mask(x) if callable(self.mask) else self.mask
            m = np.asarray(m, dtype=bool)
            if m.shape != x.shape:
                raise ValueError(f"Mask for '{self.label}' shape {m.shape} mismatch with domain {x.shape}")
            # set values outside mask to NaN (gap)
            y = np.where(m, y, np.nan)
        return y

# -----------------------------------------------------------------------------
# Structural (vertical lines)
# -----------------------------------------------------------------------------
@dataclass
class VerticalLine:
    x: float
    label: str = ''
    color: Optional[str] = None
    linestyle: str = '--'
    linewidth: float = 1.0

# -----------------------------------------------------------------------------
# Single-panel plotting
# -----------------------------------------------------------------------------

def plot_payoffs(payoffs: Sequence[Payoff], *, title: str = 'Payoff Diagram',
                 xlabel: Optional[str] = None, ylabel: str = 'Payoff',
                 vlines: Optional[Sequence[VerticalLine]] = None,
                 x_min: Optional[float] = None, x_max: Optional[float] = None,
                 y_min: Optional[float] = None, y_max: Optional[float] = None,
                 axis_color: str = 'black', axis_width: float = 1.2, grid: bool = True,
                 show_x_ticks: bool = True, show_y_ticks: bool = True,
                 figure: Optional[Figure] = None, ax: Optional[Axes] = None) -> Tuple[Figure, Axes]:
    if not payoffs:
        raise ValueError('No payoffs provided')
    domain = payoffs[0].domain
    for p in payoffs[1:]:
        if p.domain is not domain:
            raise ValueError('All payoffs must share the *same* RangeVar instance')
    x = domain.values

    if ax is None or figure is None:
        figure, ax = plt.subplots()

    # Plot payoffs
    for p in payoffs:
        y = p.evaluate()
        ax.plot(x, y, label=p.label, color=p.color)
        if p.highlight_x:
            for xv in p.highlight_x:
                # Only plot highlight if xv within domain range and (if masked) inside mask
                if x[0] <= xv <= x[-1]:
                    y_val = p.data(np.array([xv]))[0] if callable(p.data) else np.interp(xv, x, y, left=np.nan, right=np.nan)
                    if not np.isnan(y_val):
                        ax.plot(xv, y_val, 'o', color=p.color)

    # Vertical lines
    if vlines:
        for vl in vlines:
            ax.axvline(vl.x, color=vl.color, linestyle=vl.linestyle, linewidth=vl.linewidth, label=vl.label)

    # Axes formatting
    if x_min is None: x_min = float(x[0])
    if x_max is None: x_max = float(x[-1])
    ax.set_xlim(x_min, x_max)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    ax.axhline(0, color=axis_color, linestyle=':', linewidth=axis_width)
    ax.axvline(0, color=axis_color, linestyle=':', linewidth=axis_width)
    ax.set_title(title)
    ax.set_xlabel(xlabel or domain.name)
    ax.set_ylabel(ylabel)

    if not show_x_ticks:
        ax.set_xticks([])
    if not show_y_ticks:
        ax.set_yticks([])

    if any(p.label for p in payoffs) or (vlines and any(v.label for v in vlines)):
        ax.legend()
    if grid:
        ax.grid(True, alpha=0.25)
    return figure, ax

# -----------------------------------------------------------------------------
# Multi-panel support
# -----------------------------------------------------------------------------
@dataclass
class Panel:
    payoffs: Sequence[Payoff]
    vlines: Sequence[VerticalLine] | None = None
    title: str = ''
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None

def plot_panels(panels: Sequence[Panel], *, nrows: int, ncols: int,
                figsize: Tuple[float, float] = (12, 8), main_title: Optional[str] = None,
                share_y: bool = False) -> Tuple[Figure, np.ndarray]:
    if len(panels) > nrows * ncols:
        raise ValueError('More panels than subplot slots')
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.ravel()

    # Optional shared y-range
    if share_y:
        ymins, ymaxs = [], []
        for pnl in panels:
            for p in pnl.payoffs:
                y = p.evaluate()
                finite = y[np.isfinite(y)]
                if finite.size:
                    ymins.append(finite.min()); ymaxs.append(finite.max())
        if ymins:
            global_ymin, global_ymax = min(ymins), max(ymaxs)
        else:
            global_ymin = global_ymax = None
    else:
        global_ymin = global_ymax = None

    for i, pnl in enumerate(panels):
        ax = flat[i]
        plot_payoffs(list(pnl.payoffs), vlines=pnl.vlines, title=pnl.title, 
                     xlabel=pnl.xlabel, ylabel=pnl.ylabel or (pnl.xlabel and ''), # type: ignore
                     x_min=pnl.x_min, x_max=pnl.x_max, 
                     y_min=(global_ymin if share_y else pnl.y_min), 
                     y_max=(global_ymax if share_y else pnl.y_max), ax=ax, figure=fig)
    for ax in flat[len(panels):]:
        ax.axis('off')
    if main_title:
        fig.suptitle(main_title)
        fig.tight_layout(); fig.subplots_adjust(top=0.9)
    else:
        fig.tight_layout()
    return fig, axes