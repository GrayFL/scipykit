"""
Automatic text placement helpers for matplotlib.

Main APIs
---------
text_better(...)
    A vectorized ``ax.text`` wrapper that jointly searches for clean locations
    near one or more supplied anchors.

find_annotate_positions(...)
    Compute a joint batch layout without creating the final Text artists.

find_annotate_position(...)
    Only compute the preferred text position.

The placement search is performed in display/pixel coordinates, so it behaves
sensibly even when x/y data scales are very different.

Two text-size modes are provided:

``exact``
    Uses matplotlib's renderer to obtain the real text bounding box.

``mono_fast``
    Uses a lazily calibrated monospace metric cache.  On the first use of a
    font fallback chain it measures ASCII and CJK samples once; later calls
    estimate the text size without rendering a temporary label.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
import warnings
from typing import Mapping, Sequence, Literal

import numpy as np
from matplotlib.artist import Artist
from matplotlib.collections import Collection, PathCollection
from matplotlib.font_manager import FontProperties
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, FancyArrowPatch, Patch
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox

__all__ = [
    "text_better",
    "find_annotate_position",
    "find_annotate_positions",
    "find_anotate_position",  # backward-compatible typo alias
    "select_artists",
    "clear_mono_metric_cache",
    ]

# ---------------------------------------------------------------------------
# Monospace metric cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MonoFontMetric:
    """Measured glyph metrics at a reference fontsize and DPI."""

    ref_pt: float
    ref_dpi: float

    # Average advance width of a narrow / ASCII character.
    narrow_px: float

    # Average advance width of a CJK wide character.
    wide_px: float

    ascent_px: float
    descent_px: float

    @property
    def line_height_px(self) -> float:
        return self.ascent_px + self.descent_px

    @property
    def wide_ratio(self) -> float:
        return self.wide_px / self.narrow_px


# Internal cache.  The key describes the effective FontProperties / backend.
_MONO_METRIC_CACHE: dict[tuple, _MonoFontMetric] = {}

# A long-ish sample averages out small rasterization / hinting differences.
_ASCII_CALIBRATION_TEXT = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    ".,:;+-=*/()[]{}<>_@#$%"
    )

_CJK_CALIBRATION_TEXT = "中文测试字体宽度天地玄黄宇宙洪荒"


def clear_mono_metric_cache() -> None:
    """Clear all lazily calibrated ``mono_fast`` font metrics."""
    _MONO_METRIC_CACHE.clear()


def _get_renderer(fig):
    """
    Return a matplotlib renderer.

    Most common canvases expose ``get_renderer`` directly.  If the renderer is
    not ready yet, a single draw is performed.
    """
    try:
        return fig.canvas.get_renderer()
    except Exception:
        fig.canvas.draw()
        return fig.canvas.get_renderer()


def _make_metric_key(fig, text_artist) -> tuple:
    """
    Build a cache key for the effective font environment.

    Keeping the complete fallback family list is important for setups such as
    ``["Inconsolata", "Sarasa Mono SC"]`` where ASCII and CJK glyphs may be
    supplied by different fonts.
    """
    prop = text_artist.get_fontproperties()

    return (
        tuple(prop.get_family()),
        str(prop.get_style()),
        str(prop.get_weight()),
        str(prop.get_stretch()),
        str(prop.get_variant()),
        prop.get_file(),
        type(fig.canvas).__module__,
        type(fig.canvas).__name__,
        )


def _calibrate_mono_metric(
    fig,
    text_artist,
    *,
    ref_pt: float = 12.0,
    verify: bool = True,
    ) -> _MonoFontMetric:
    """
    Measure ASCII and CJK glyph widths once for the current font fallback chain.
    """
    renderer = _get_renderer(fig)

    prop = text_artist.get_fontproperties().copy()
    prop.set_size(ref_pt)

    # Narrow / ASCII width.
    w_ascii, _, _ = renderer.get_text_width_height_descent(
        _ASCII_CALIBRATION_TEXT,
        prop,
        ismath=False,
    )
    narrow_px = w_ascii / len(_ASCII_CALIBRATION_TEXT)

    # Wide / CJK width.  Matplotlib performs the normal fallback selection.
    w_cjk, _, _ = renderer.get_text_width_height_descent(
        _CJK_CALIBRATION_TEXT,
        prop,
        ismath=False,
    )
    wide_px = w_cjk / len(_CJK_CALIBRATION_TEXT)

    # Vertical metrics: keep the larger ASCII/CJK extent.
    _, h_ascii, d_ascii = renderer.get_text_width_height_descent(
        "Mgjpq",
        prop,
        ismath=False,
    )
    _, h_cjk, d_cjk = renderer.get_text_width_height_descent(
        "中文测试",
        prop,
        ismath=False,
    )

    ascent_px = max(h_ascii - d_ascii, h_cjk - d_cjk)
    descent_px = max(d_ascii, d_cjk)

    metric = _MonoFontMetric(
        ref_pt=float(ref_pt),
        ref_dpi=float(fig.dpi),
        narrow_px=float(narrow_px),
        wide_px=float(wide_px),
        ascent_px=float(ascent_px),
        descent_px=float(descent_px),
        )

    if verify:
        _verify_mono_metric(renderer, prop, metric)

    return metric


def _verify_mono_metric(
    renderer,
    prop: FontProperties,
    metric: _MonoFontMetric,
    *,
    mono_tol: float = 0.08,
    wide_ratio_tol: float = 0.15,
    ) -> None:
    """Warn when the active font setup does not look like a CJK monospace pair."""
    widths = []

    for ch in ("i", "M", "0", "W", ".", "@"):
        w, _, _ = renderer.get_text_width_height_descent(
            ch,
            prop,
            ismath=False,
        )
        widths.append(w)

    widths = np.asarray(widths, dtype=float)
    spread = (widths.max() - widths.min()) / max(metric.narrow_px, 1e-12)

    if spread > mono_tol:
        warnings.warn(
            "mono_fast: the ASCII font does not appear to be monospaced "
            f"(relative width spread={spread:.3f}).",
            RuntimeWarning,
            stacklevel=4,
            )

    if abs(metric.wide_ratio - 2.0) > wide_ratio_tol:
        warnings.warn(
            "mono_fast: the measured CJK/ASCII width ratio is "
            f"{metric.wide_ratio:.3f}, not approximately 2. "
            "The measured ratio will still be used.",
            RuntimeWarning,
            stacklevel=4,
            )


def _get_mono_metric(
    fig,
    text_artist,
    *,
    ref_pt: float = 12.0,
    verify: bool = True,
    ) -> _MonoFontMetric:
    """Return a cached metric, lazily calibrating it on first use."""
    key = _make_metric_key(fig, text_artist)

    metric = _MONO_METRIC_CACHE.get(key)
    if metric is None:
        metric = _calibrate_mono_metric(
            fig,
            text_artist,
            ref_pt=ref_pt,
            verify=verify,
            )
        _MONO_METRIC_CACHE[key] = metric

    return metric


# ---------------------------------------------------------------------------
# Fast text size estimation
# ---------------------------------------------------------------------------

_BOX_PAD_PATTERN = re.compile(r"(?:^|,)\s*pad\s*=\s*([0-9.+\-eE]+)")

_FAST_SUPPORTED_BOXSTYLES = {
    "square",
    "round",
    "round4",
    }


def _east_asian_is_wide(ch: str) -> bool:
    """Return True for East Asian wide/full-width characters."""
    return unicodedata.east_asian_width(ch) in ("W", "F")


def _line_width_fast(
    line: str,
    metric: _MonoFontMetric,
    scale: float,
    *,
    tabsize: int = 4,
    ) -> float:
    """
    Estimate one line's width.

    Narrow characters use the measured ASCII advance; East-Asian W/F
    characters use the measured CJK advance.  Combining marks consume no width.
    """
    narrow = metric.narrow_px * scale
    wide = metric.wide_px * scale

    width = 0.0
    logical_col = 0

    for ch in line:
        if unicodedata.combining(ch):
            continue

        if ch == "\t":
            n = tabsize - logical_col%tabsize
            width += n * narrow
            logical_col += n
            continue

        if _east_asian_is_wide(ch):
            width += wide
            logical_col += 2
        else:
            width += narrow
            logical_col += 1

    return width


def _text_size_fast_px(
    fig,
    text_artist,
    text: str,
    metric: _MonoFontMetric,
    *,
    tabsize: int = 4,
    ) -> tuple[float, float, float, float]:
    """Estimate text width, height, ascent and descent in pixels."""
    scale = (
        text_artist.get_fontsize() / metric.ref_pt * fig.dpi
        / metric.ref_dpi
        )

    lines = str(text).split("\n")

    widths = [
        _line_width_fast(
            line,
            metric,
            scale,
            tabsize=tabsize,
            ) for line in lines
        ]

    width = max(widths, default=0.0)

    ascent = metric.ascent_px * scale
    descent = metric.descent_px * scale
    line_height = ascent + descent

    # Matplotlib 3.11 exposes get_linespacing() and defaults to 'normal'.
    spacing = (text_artist.get_linespacing() if hasattr(text_artist, 'get_linespacing')
               else getattr(text_artist, '_linespacing', 1.2))
    linespacing = 1.2 if spacing == 'normal' else float(spacing)
    n_lines = max(1, len(lines))

    if n_lines == 1:
        height = line_height
    else:
        # Good approximation for ordinary multiline monospace labels.
        baseline_step = ascent * linespacing
        height = line_height + (n_lines-1) * baseline_step

    return width, height, ascent, descent


def _parse_boxstyle_name(boxstyle) -> str | None:
    if not isinstance(boxstyle, str):
        return None
    return boxstyle.split(",", 1)[0].strip().lower()


def _bbox_extra_px(
    fig,
    text_artist,
    bbox_kwargs: dict | None,
    ) -> tuple[float, float]:
    """
    Estimate symmetric extra x/y extent introduced by a normal text bbox.

    For unusual shape-changing boxstyles, ``mono_fast`` should preferably
    fall back to exact mode.
    """
    if not bbox_kwargs:
        return 0.0, 0.0

    fontsize_px = text_artist.get_fontsize() * fig.dpi / 72.0

    boxstyle = bbox_kwargs.get("boxstyle", "square,pad=0.3")
    pad = 0.3

    if isinstance(boxstyle, str):
        match = _BOX_PAD_PATTERN.search(boxstyle)
        if match:
            pad = float(match.group(1))

    pad_px = pad * fontsize_px

    linewidth_pt = float(
        bbox_kwargs.get(
            "linewidth",
            bbox_kwargs.get("lw", 1.0),
            )
        )
    linewidth_px = linewidth_pt * fig.dpi / 72.0

    extra = pad_px + linewidth_px/2.0
    return extra, extra


def _bbox_from_size_and_alignment(
    text_artist,
    width: float,
    height: float,
    descent: float,
    ) -> np.ndarray:
    """
    Convert width/height into a bbox relative to the text anchor.

    Returns ``[x0, y0, x1, y1]`` in display pixels.
    """
    ha = text_artist.get_horizontalalignment()
    va = text_artist.get_verticalalignment()

    if ha == "center":
        x0 = -width / 2.0
    elif ha == "right":
        x0 = -width
    else:
        x0 = 0.0
    x1 = x0 + width

    if va == "top":
        y0 = -height
    elif va == "center":
        y0 = -height / 2.0
    elif va == "bottom":
        y0 = 0.0
    elif va == "center_baseline":
        y0 = -descent - 0.5 * max(height - descent, 0.0)
    else:  # baseline and uncommon values
        y0 = -descent
    y1 = y0 + height

    # Approximate rotation by rotating the four corners and taking the enclosing
    # rectangle.  Exact rotation_mode semantics belong to exact mode.
    angle = text_artist.get_rotation()
    try:
        angle = float(angle)
    except (TypeError, ValueError):
        angle = 0.0

    if angle % 360:
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)

        corners = np.array(
            [
                [x0, y0],
                [x0, y1],
                [x1, y0],
                [x1, y1],
                ],
            dtype=float,
            )
        rot = np.array([[c, -s], [s, c]])
        corners = corners @ rot.T

        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)

    return np.array([x0, y0, x1, y1], dtype=float)


def _fast_bbox_relative_px(
    fig,
    text_artist,
    text: str,
    *,
    bbox_kwargs: dict | None = None,
    tabsize: int = 4,
    verify_metric: bool = True,
    ) -> np.ndarray:
    """Estimate the text+bbox rectangle relative to its anchor in pixels."""
    metric = _get_mono_metric(
        fig,
        text_artist,
        verify=verify_metric,
        )

    width, height, _, descent = _text_size_fast_px(
        fig,
        text_artist,
        text,
        metric,
        tabsize=tabsize,
    )

    extra_x, extra_y = _bbox_extra_px(
        fig,
        text_artist,
        bbox_kwargs,
    )

    width += 2.0 * extra_x
    height += 2.0 * extra_y
    descent += extra_y

    return _bbox_from_size_and_alignment(
        text_artist,
        width,
        height,
        descent,
        )


# ---------------------------------------------------------------------------
# Exact text box measurement
# ---------------------------------------------------------------------------


def _exact_bbox_relative_px(
    fig,
    ax,
    anchor,
    text: str,
    fontdict: dict | None,
    text_kwargs: dict,
    ) -> np.ndarray:
    """
    Measure the real text/bbox extent with matplotlib's renderer.

    A temporary Text artist is created, measured, then immediately removed.
    """
    renderer = _get_renderer(fig)

    probe = ax.text(
        anchor[0],
        anchor[1],
        text,
        fontdict=fontdict,
        **text_kwargs,
        )

    try:
        transform = probe.get_transform()
        anchor_px = transform.transform(anchor)

        boxes = [probe.get_window_extent(renderer)]

        patch = probe.get_bbox_patch()
        if patch is not None:
            probe.update_bbox_position_size(renderer)
            boxes.append(patch.get_window_extent(renderer))

        bb = Bbox.union(boxes)

        return np.array(
            [
                bb.x0 - anchor_px[0],
                bb.y0 - anchor_px[1],
                bb.x1 - anchor_px[0],
                bb.y1 - anchor_px[1],
                ],
            dtype=float,
            )
    finally:
        probe.remove()


# ---------------------------------------------------------------------------
# Obstacle geometry and artist selection
# ---------------------------------------------------------------------------


@dataclass
class _ObstacleGeometry:
    """Obstacle primitives expressed in display/pixel coordinates."""

    points: np.ndarray
    bboxes: np.ndarray


_DEFAULT_ARTIST_TYPES = (Line2D, Patch, Text, Collection)

_ARTIST_TYPE_ALIASES = {
    "line": Line2D,
    "lines": Line2D,
    "patch": Patch,
    "patches": Patch,
    "text": Text,
    "texts": Text,
    "collection": Collection,
    "collections": Collection,
    "scatter": PathCollection,
    "arrow": (FancyArrow, FancyArrowPatch),
    "arrows": (FancyArrow, FancyArrowPatch),
    "image": AxesImage,
    "images": AxesImage,
    }


def _artist_label_matches(artist: Artist, patterns) -> bool:
    """Match an artist label with one or more regular expressions."""
    if isinstance(patterns, (str, re.Pattern)):
        patterns = [patterns]
    label = str(artist.get_label() or "")
    try:
        return any(
            re.search(pattern, label) is not None for pattern in patterns
            )
    except re.error as exc:
        raise ValueError(
            f"Invalid label regular expression: {exc}"
            ) from exc


def _artist_type_names(artist: Artist) -> set[str]:
    """Return short and fully-qualified names from an artist's MRO."""
    names = set()
    for artist_class in type(artist).__mro__:
        names.add(artist_class.__name__)
        names.add(artist_class.__qualname__)
        names.add(f"{artist_class.__module__}.{artist_class.__qualname__}")
    return names


def _artist_type_matches(artist: Artist, type_spec) -> bool:
    """Match a class, class-name string, or a collection mixing both."""
    if isinstance(type_spec, type):
        return isinstance(artist, type_spec)
    if isinstance(type_spec, str):
        requested = type_spec.casefold()
        alias = _ARTIST_TYPE_ALIASES.get(requested)
        if alias is not None:
            return isinstance(artist, alias)
        return any(
            name.casefold() == requested
            for name in _artist_type_names(artist)
            )
    try:
        return any(
            _artist_type_matches(artist, item) for item in type_spec
            )
    except TypeError as exc:
        raise TypeError(
            "artist type selectors must be classes, class-name strings, "
            "or iterables mixing both"
            ) from exc


def _known_artist_type_name(name: str) -> bool:
    """Whether a string names any currently imported Artist subclass."""
    requested = name.casefold()
    if requested in _ARTIST_TYPE_ALIASES:
        return True
    pending = [Artist]
    seen = set()
    while pending:
        artist_class = pending.pop()
        if artist_class in seen:
            continue
        seen.add(artist_class)
        names = {
            artist_class.__name__,
            artist_class.__qualname__,
            f"{artist_class.__module__}.{artist_class.__qualname__}",
            }
        if any(item.casefold() == requested for item in names):
            return True
        pending.extend(artist_class.__subclasses__())
    return False


def _matches_artist_selector(artist: Artist, selector) -> bool:
    """Implement one selector; iterables of selectors are treated as a union."""
    if selector is None or selector is False:
        return False
    if selector is True:
        return True
    if isinstance(selector, Artist):
        return artist is selector
    if isinstance(selector, type):
        return isinstance(artist, selector)
    if isinstance(selector, str):
        if selector == "all":
            return True
        if _known_artist_type_name(selector):
            return _artist_type_matches(artist, selector)
        return _artist_label_matches(artist, selector)
    if isinstance(selector, re.Pattern):
        return _artist_label_matches(artist, selector)
    if isinstance(selector, Mapping):
        artist_type = selector.get(
            "type",
            selector.get(
                "class",
                selector.get("artist_class", selector.get("artist_type")),
                ),
            )
        if artist_type is not None and not _artist_type_matches(artist,
                                                                artist_type
                                                               ):
            return False
        if "label" in selector and not _artist_label_matches(
                artist, selector["label"]
            ):
            return False
        if "visible" in selector and bool(artist.get_visible()
                                         ) != bool(selector["visible"]):
            return False
        predicate = selector.get("predicate")
        if predicate is not None and not bool(predicate(artist)):
            return False
        return True
    if callable(selector):
        return bool(selector(artist))
    try:
        return any(
            _matches_artist_selector(artist, item) for item in selector
            )
    except TypeError as exc:
        raise TypeError(
            f"Unsupported artist selector: {selector!r}"
            ) from exc


def select_artists(ax, selector=None) -> list[Artist]:
    """
    Select artists below ``ax`` by label, class, predicate, or their union.

    ``selector=None`` selects the common obstacle types: lines, collections,
    patches, and text.  Artist categories may be classes, class-name strings,
    common aliases, or lists mixing them, such as
    ``[Line2D, "Patch", "text", "scatter"]``.  Supported aliases include
    ``line``, ``patch``, ``text``, ``collection``, ``scatter``, ``arrow`` and
    ``image`` (plus plural forms).  A dictionary combines filters with AND semantics;
    ``label`` values are regular
    expressions, for example
    ``{"type": ["Line2D", "PathCollection"], "label": r"^signal-\\d+$"}``.
    A top-level string that names an Artist class is a type selector; other
    strings are treated as label regular expressions.  Images are not part of
    the default selection.
    """
    if selector is None:
        selector = {"type": _DEFAULT_ARTIST_TYPES}

    candidates = list(ax.findobj(include_self=False))
    # Annotation owns its arrow patch without exposing it through get_children,
    # so ax.findobj() cannot discover the arrow by itself.
    for artist in tuple(candidates):
        if isinstance(artist,
                        Annotation) and artist.arrow_patch is not None:
            candidates.append(artist.arrow_patch)

    result = []
    seen = set()
    for artist in candidates:
        identity = id(artist)
        if identity in seen or artist is ax.patch or not artist.get_visible(
        ):
            continue
        if _matches_artist_selector(artist, selector):
            result.append(artist)
            seen.add(identity)
    return result


def _densify_path_px(points_px, step_px: float = 4.0) -> np.ndarray:
    """Densify a finite polyline in display space."""
    points_px = np.asarray(points_px, dtype=float)
    if len(points_px) < 2:
        return points_px

    result = [points_px[0]]
    for p0, p1 in zip(points_px[:-1], points_px[1:]):
        distance = np.linalg.norm(p1 - p0)
        n = max(1, int(np.ceil(distance / max(float(step_px), 1e-6))))
        fraction = np.linspace(0.0, 1.0, n + 1)[1:, None]
        result.extend(p0 + fraction * (p1-p0))
    return np.asarray(result, dtype=float)


def _finite_path_chunks(points: np.ndarray) -> list[np.ndarray]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("paths must have shape (N, 2)")
    finite = np.isfinite(points).all(axis=1)
    if finite.all():
        return [points] if len(points) else []
    boundaries = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    return [
        points[start:stop] for start, stop in boundaries.reshape(-1, 2)
        ]


def _distance_points_to_bbox_border(
    points: np.ndarray,
    bb: np.ndarray,
    ) -> np.ndarray:
    """Euclidean distance from points to the rectangle boundary."""
    x0, y0, x1, y1 = bb
    dx = np.maximum(np.maximum(x0 - points[:, 0], 0.0), points[:, 0] - x1)
    dy = np.maximum(np.maximum(y0 - points[:, 1], 0.0), points[:, 1] - y1)
    distances = np.hypot(dx, dy)

    inside = (dx == 0.0) & (dy == 0.0)
    if np.any(inside):
        inside_points = points[inside]
        distances[inside] = np.min(
            np.column_stack((
                inside_points[:, 0] - x0,
                x1 - inside_points[:, 0],
                inside_points[:, 1] - y0,
                y1 - inside_points[:, 1],
                )),
            axis=1,
            )
    return distances


def _distance_point_to_bbox_borders(
    point: np.ndarray,
    bboxes: np.ndarray,
    ) -> np.ndarray:
    """Euclidean distance from one point to every rectangle boundary."""
    point = np.asarray(point, dtype=float)
    dx = np.maximum(
        np.maximum(bboxes[:, 0] - point[0], 0.0), point[0] - bboxes[:, 2]
        )
    dy = np.maximum(
        np.maximum(bboxes[:, 1] - point[1], 0.0), point[1] - bboxes[:, 3]
        )
    distances = np.hypot(dx, dy)
    inside = (dx == 0.0) & (dy == 0.0)
    if np.any(inside):
        inside_boxes = bboxes[inside]
        distances[inside] = np.min(
            np.column_stack((
                point[0] - inside_boxes[:, 0],
                inside_boxes[:, 2] - point[0],
                point[1] - inside_boxes[:, 1],
                inside_boxes[:, 3] - point[1],
                )),
            axis=1,
            )
    return distances


def _distance_bbox_borders_to_bbox(
    bboxes: np.ndarray,
    bb: np.ndarray,
    ) -> np.ndarray:
    """Euclidean shortest distance between axis-aligned rectangle borders."""
    dx = np.maximum(
        np.maximum(bb[0] - bboxes[:, 2], 0.0), bboxes[:, 0] - bb[2]
        )
    dy = np.maximum(
        np.maximum(bb[1] - bboxes[:, 3], 0.0), bboxes[:, 1] - bb[3]
        )
    distances = np.hypot(dx, dy)

    # When rectangles overlap, their boundaries normally intersect (distance
    # zero).  Strict containment is the exception: the nearest borders are
    # separated by the smallest of the four enclosing margins.
    bb_contains = ((bb[0] <= bboxes[:, 0]) & (bb[1] <= bboxes[:, 1])
                    & (bb[2] >= bboxes[:, 2]) & (bb[3] >= bboxes[:, 3]))
    if np.any(bb_contains):
        contained = bboxes[bb_contains]
        distances[bb_contains] = np.min(
            np.column_stack((
                contained[:, 0] - bb[0],
                bb[2] - contained[:, 2],
                contained[:, 1] - bb[1],
                bb[3] - contained[:, 3],
                )),
            axis=1,
            )

    bbox_contains_bb = ((bboxes[:, 0] <= bb[0]) & (bboxes[:, 1] <= bb[1])
                        & (bboxes[:, 2] >= bb[2]) &
                        (bboxes[:, 3] >= bb[3]))
    # Equal rectangles satisfy both masks and correctly retain distance zero.
    bbox_contains_bb &= ~bb_contains
    if np.any(bbox_contains_bb):
        containers = bboxes[bbox_contains_bb]
        distances[bbox_contains_bb] = np.min(
            np.column_stack((
                bb[0] - containers[:, 0],
                containers[:, 2] - bb[2],
                bb[1] - containers[:, 1],
                containers[:, 3] - bb[3],
                )),
            axis=1,
            )
    return distances


def _normalize_paths(avoid_paths) -> list[np.ndarray]:
    if avoid_paths is None:
        return []
    try:
        array = np.asarray(avoid_paths, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is not None and array.ndim == 2 and array.shape[1] == 2:
        return [array]
    return [np.asarray(path, dtype=float) for path in avoid_paths]


def _valid_bbox_array(bb) -> np.ndarray | None:
    values = np.asarray([bb.x0, bb.y0, bb.x1, bb.y1], dtype=float)
    if not np.isfinite(values).all(
    ) or values[2] < values[0] or values[3] < values[1]:
        return None
    return values


def _geometry_from_artists(
        fig, ax, artists, step_px: float
    ) -> _ObstacleGeometry:
    renderer = _get_renderer(fig)
    point_chunks = []
    bboxes = []

    selected_ids = {id(artist) for artist in artists}
    for annotation in ax.findobj(
            match=lambda item: isinstance(item, Annotation),
            include_self=False
        ):
        if (annotation.arrow_patch is not None
                and id(annotation.arrow_patch) in selected_ids):
            # This initializes the internal patch even before the first canvas
            # draw; otherwise its path still sits near (0, 0) display pixels.
            annotation.update_positions(renderer)

    for artist in artists:
        try:
            if isinstance(artist, Line2D):
                vertices = artist.get_transform().transform_path(
                    artist.get_path()
                    ).vertices
                for chunk in _finite_path_chunks(vertices):
                    if len(chunk):
                        point_chunks.append(
                            _densify_path_px(chunk, step_px)
                            )
                continue

            if isinstance(artist, FancyArrowPatch):
                path = artist.get_transform().transform_path(
                    artist.get_path()
                    )
                # Interpolate Bezier arrow shafts/heads before ordinary pixel
                # densification so curved arrows are represented faithfully.
                vertices = path.interpolated(steps=8).vertices
                for chunk in _finite_path_chunks(vertices):
                    if len(chunk):
                        point_chunks.append(
                            _densify_path_px(chunk, step_px)
                            )
                continue

            if isinstance(artist, PathCollection):
                offsets = np.asarray(artist.get_offsets(), dtype=float)
                if offsets.ndim == 2 and offsets.shape[
                        1] == 2 and len(offsets):
                    offsets = artist.get_offset_transform(
                    ).transform(offsets)
                    offsets = offsets[np.isfinite(offsets).all(axis=1)]
                    if len(offsets):
                        sizes = np.asarray(artist.get_sizes(), dtype=float)
                        radius = (
                            0.6 * np.sqrt(float(np.nanmax(sizes)))
                            * fig.dpi / 72.0 if sizes.size else 0.0
                            )
                        if radius > 0:
                            bboxes.extend(
                                np.column_stack((
                                    offsets[:, 0] - radius,
                                    offsets[:, 1] - radius,
                                    offsets[:, 0] + radius,
                                    offsets[:, 1] + radius,
                                    ))
                                )
                        else:
                            point_chunks.append(offsets)
                continue

            # Filled patches and existing text are best represented by their
            # complete extent, rather than just sparse vertices.
            if isinstance(artist, (Patch, Text)):
                if isinstance(artist, Text):
                    if not artist.get_text():
                        continue
                    extents = [artist.get_window_extent(renderer)]
                    bbox_patch = artist.get_bbox_patch()
                    if bbox_patch is not None:
                        artist.update_bbox_position_size(renderer)
                        extents.append(
                            bbox_patch.get_window_extent(renderer)
                            )
                    extent = Bbox.union(extents)
                else:
                    extent = artist.get_window_extent(renderer)
                bb = _valid_bbox_array(extent)
                if bb is not None and (bb[2] > bb[0] or bb[3] > bb[1]):
                    bboxes.append(bb)
                continue

            if isinstance(artist, Collection):
                data_bb = artist.get_datalim(ax.transData)
                if np.isfinite(data_bb.extents).all():
                    bb = _valid_bbox_array(
                        Bbox(ax.transData.transform(data_bb.get_points()))
                        )
                    if bb is not None:
                        bboxes.append(bb)
                continue

            bb = _valid_bbox_array(artist.get_window_extent(renderer))
            if bb is not None and (bb[2] > bb[0] or bb[3] > bb[1]):
                bboxes.append(bb)
        except (AttributeError, TypeError, ValueError):
            # Some third-party artists expose an incomplete extent API.  One
            # unmeasurable artist should not make annotation placement fail.
            continue

    points = (
        np.vstack(point_chunks)
        if point_chunks else np.empty((0, 2), dtype=float)
        )
    bbox_array = (
        np.asarray(bboxes, dtype=float).reshape(-1, 4)
        if bboxes else np.empty((0, 4), dtype=float)
        )
    return _ObstacleGeometry(points=points, bboxes=bbox_array)


def _collect_obstacle_geometry(
    fig,
    ax,
    *,
    avoid_points,
    avoid_paths,
    avoid_transform,
    avoid_artists,
    allow_artists,
    auto_avoid,
    path_sample_step_px,
    ) -> _ObstacleGeometry:
    if avoid_transform is None:
        avoid_transform = ax.transData

    point_chunks = []
    if avoid_points is not None:
        points = np.asarray(avoid_points, dtype=float)
        if points.size:
            points = np.atleast_2d(points)
            if points.shape[1] != 2:
                raise ValueError("avoid_points must have shape (N, 2)")
            points = avoid_transform.transform(points)
            point_chunks.append(points[np.isfinite(points).all(axis=1)])

    for path in _normalize_paths(avoid_paths):
        for chunk in _finite_path_chunks(path):
            transformed = avoid_transform.transform(chunk)
            point_chunks.append(
                _densify_path_px(transformed, path_sample_step_px)
                )

    manual_supplied = avoid_points is not None or avoid_paths is not None
    use_default = (not manual_supplied
                  ) if auto_avoid is None else bool(auto_avoid)
    if avoid_artists is False:
        selected = []
    elif avoid_artists is None:
        selected = select_artists(ax) if use_default else []
    else:
        selected = select_artists(ax, avoid_artists)

    if allow_artists is not None and selected:
        allowed_ids = set()
        for item in select_artists(ax, allow_artists):
            allowed_ids.update(
                id(descendant)
                for descendant in item.findobj(include_self=True)
                )
            if isinstance(item,
                            Annotation) and item.arrow_patch is not None:
                allowed_ids.add(id(item.arrow_patch))
        selected = [
            item for item in selected if id(item) not in allowed_ids
            ]

    artist_geometry = _geometry_from_artists(
        fig, ax, selected, path_sample_step_px
        )
    if len(artist_geometry.points):
        point_chunks.append(artist_geometry.points)

    points = (
        np.vstack(point_chunks)
        if point_chunks else np.empty((0, 2), dtype=float)
        )
    return _ObstacleGeometry(points=points, bboxes=artist_geometry.bboxes)


# ---------------------------------------------------------------------------
# Batch normalization and global placement search
# ---------------------------------------------------------------------------

_PER_ITEM_TEXT_KWARGS = {
    "alpha",
    "animated",
    "bbox",
    "c",
    "clip_on",
    "color",
    "family",
    "fontfamily",
    "fontname",
    "fontproperties",
    "fontsize",
    "fontstretch",
    "fontstyle",
    "fontvariant",
    "fontweight",
    "gid",
    "ha",
    "label",
    "linespacing",
    "ma",
    "multialignment",
    "picker",
    "rasterized",
    "rotation",
    "rotation_mode",
    "size",
    "stretch",
    "style",
    "url",
    "va",
    "variant",
    "verticalalignment",
    "visible",
    "weight",
    "wrap",
    "zorder",
    }


def _is_scalar_input(value) -> bool:
    return isinstance(value, str) or np.asarray(value).ndim == 0


def _normalize_text_batch(x, y, text):
    try:
        x_values, y_values, text_values = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(text, dtype=object),
            )
    except ValueError as exc:
        raise ValueError(
            "x, y, and t must be scalar or broadcastable arrays"
            ) from exc
    anchors = np.column_stack((x_values.ravel(), y_values.ravel()))
    texts = [str(item) for item in text_values.ravel()]
    return anchors, texts


def _style_is_single_value(key: str, value) -> bool:
    if isinstance(value, (str, bytes, Mapping, FontProperties)):
        return True
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return True
    if key in {"color", "c"}:
        from matplotlib.colors import is_color_like
        if is_color_like(value):
            return True
    if key in {"family", "fontfamily"} and isinstance(value,
                                                        (list, tuple)):
        return all(isinstance(item, str) for item in value)
    return not isinstance(value, (list, tuple, np.ndarray))


def _broadcast_style(value, n: int, key: str) -> list:
    if n == 1 or _style_is_single_value(key, value):
        return [value] * n
    values = list(value)
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(
            f"text style {key!r} has length {len(values)}; expected 1 or {n}"
            )
    return values


def _split_text_kwargs(text_kwargs: Mapping | None, n: int) -> list[dict]:
    kwargs = dict(text_kwargs or {})
    result = [dict() for _ in range(n)]
    for key, value in kwargs.items():
        values = (
            _broadcast_style(value, n, key)
            if key in _PER_ITEM_TEXT_KWARGS else [value] * n
            )
        for item, item_value in zip(result, values):
            item[key] = item_value
    return result


def _split_fontdict(fontdict, n: int) -> list[dict | None]:
    if fontdict is None or isinstance(fontdict, Mapping):
        return [fontdict] * n
    values = list(fontdict)
    if len(values) == 1:
        values *= n
    if len(values
          ) != n or not all(item is None or isinstance(item, Mapping)
                            for item in values):
        raise ValueError(
            "fontdict must be a mapping or a sequence of mappings"
            )
    return values


def _apply_default_alignments(item_kwargs, item_fontdicts) -> None:
    for kwargs, fontdict in zip(item_kwargs, item_fontdicts):
        fontdict = fontdict or {}
        if ("ha" not in kwargs and "horizontalalignment" not in kwargs
                and "ha" not in fontdict
                and "horizontalalignment" not in fontdict):
            kwargs["ha"] = "center"
        if ("va" not in kwargs and "verticalalignment" not in kwargs
                and "va" not in fontdict
                and "verticalalignment" not in fontdict):
            kwargs["va"] = "center"


def _measure_relative_bbox(
    fig,
    ax,
    anchor,
    text,
    fontdict,
    text_kwargs,
    *,
    mode,
    fast_fallback,
    tabsize,
    verify_metric,
    ):
    probe = ax.text(
        anchor[0], anchor[1], text, fontdict=fontdict, **text_kwargs
        )
    try:
        transform = probe.get_transform()
        anchor_px = transform.transform(anchor)
        chosen_mode = mode.lower()
        if chosen_mode == "mono_fast" and text_artist_uses_complex_layout(
                probe, text_kwargs
            ):
            if fast_fallback == "exact":
                chosen_mode = "exact"
            elif fast_fallback == "raise":
                raise ValueError(
                    "mono_fast cannot safely estimate this text configuration"
                    )
            else:
                raise ValueError(
                    "fast_fallback must be 'exact' or 'raise'"
                    )

        if chosen_mode == "mono_fast":
            relative = _fast_bbox_relative_px(
                fig,
                probe,
                text,
                bbox_kwargs=text_kwargs.get("bbox"),
                tabsize=tabsize,
                verify_metric=verify_metric,
                )
        elif chosen_mode == "exact":
            probe.remove()
            probe = None
            relative = _exact_bbox_relative_px(
                fig, ax, anchor, text, fontdict, text_kwargs
                )
        else:
            raise ValueError("mode must be 'exact' or 'mono_fast'")
    finally:
        if probe is not None:
            probe.remove()
    return relative, transform, np.asarray(anchor_px, dtype=float)


def _candidate_positions(anchor_px, relative_bbox, radii_px, n_angles):
    width = relative_bbox[2] - relative_bbox[0]
    height = relative_bbox[3] - relative_bbox[1]
    diagonal = max(float(np.hypot(width, height)), 1.0)
    if radii_px is None:
        first = max(16.0, 0.25 * diagonal)
        radii = (
            first, first + 16.0, first + 32.0, first + 52.0, first + 76.0
            )
    else:
        radii = tuple(
            float(value) for value in radii_px if float(value) > 0
            )
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    offsets = [np.zeros((1, 2), dtype=float)]

    # If the text anchor lies inside its relative box (the normal centered
    # case), add one analytically computed candidate per direction whose box
    # border passes exactly through the original anchor.  Radius rings remain
    # available for obstacle avoidance and wider layouts.
    if (relative_bbox[0] <= 0.0 <= relative_bbox[2]
            and relative_bbox[1] <= 0.0 <= relative_bbox[3]
            and len(directions)):
        ray = -directions
        tx = np.full(len(ray), np.inf, dtype=float)
        ty = np.full(len(ray), np.inf, dtype=float)
        positive_x = ray[:, 0] > 1e-12
        negative_x = ray[:, 0] < -1e-12
        positive_y = ray[:, 1] > 1e-12
        negative_y = ray[:, 1] < -1e-12
        tx[positive_x] = relative_bbox[2] / ray[positive_x, 0]
        tx[negative_x] = relative_bbox[0] / ray[negative_x, 0]
        ty[positive_y] = relative_bbox[3] / ray[positive_y, 1]
        ty[negative_y] = relative_bbox[1] / ray[negative_y, 1]
        touch_radii = np.minimum(tx, ty)
        valid = np.isfinite(touch_radii) & (touch_radii > 1e-9)
        if np.any(valid):
            offsets.append(touch_radii[valid, None] * directions[valid])

    offsets.extend(float(radius) * directions for radius in radii)
    positions = anchor_px + np.vstack(offsets)
    boxes = positions[:, [0, 1, 0, 1]] + relative_bbox
    return positions, boxes, diagonal


def _obstacle_costs(
    boxes,
    anchor_px,
    geometry,
    *,
    ignore_anchor_px,
    safe_distance_px,
    collision_weight,
    proximity_weight,
    ):
    costs = np.zeros(len(boxes), dtype=float)
    points = geometry.points
    if len(points) and ignore_anchor_px > 0:
        points = points[np.linalg.norm(points - anchor_px, axis=1) >
                        ignore_anchor_px]

    for index, bb in enumerate(boxes):
        if len(points):
            inside = ((points[:, 0] >= bb[0]) & (points[:, 0] <= bb[2])
                        & (points[:, 1] >= bb[1]) &
                        (points[:, 1] <= bb[3]))
            distances = _distance_points_to_bbox_border(points, bb)
            costs[index] += collision_weight * np.count_nonzero(inside)
            if safe_distance_px > 0:
                near = np.clip(safe_distance_px - distances, 0.0, None)
                costs[index] += proximity_weight * np.sum(
                    (near / safe_distance_px)**2
                    )

        if len(geometry.bboxes):
            other = geometry.bboxes
            overlap_w = np.clip(
                np.minimum(bb[2], other[:, 2])
                - np.maximum(bb[0], other[:, 0]),
                0.0,
                None,
                )
            overlap_h = np.clip(
                np.minimum(bb[3], other[:, 3])
                - np.maximum(bb[1], other[:, 1]),
                0.0,
                None,
                )
            overlap = overlap_w * overlap_h
            hit = overlap > 0
            if np.any(hit):
                own_area = max((bb[2] - bb[0]) * (bb[3] - bb[1]), 1.0)
                costs[index] += collision_weight * np.sum(
                    1.0 + overlap[hit] / own_area
                    )
            if safe_distance_px > 0:
                distances = _distance_bbox_borders_to_bbox(other, bb)
                near = np.clip(safe_distance_px - distances, 0.0, None)
                costs[index] += proximity_weight * np.sum(
                    (near / safe_distance_px)**2
                    )
    return costs


def _pair_cost_vector(
    boxes,
    other,
    *,
    collision_weight,
    proximity_weight,
    safe_distance_px,
    ):
    overlap_w = np.clip(
        np.minimum(boxes[:, 2], other[2])
        - np.maximum(boxes[:, 0], other[0]),
        0.0,
        None,
        )
    overlap_h = np.clip(
        np.minimum(boxes[:, 3], other[3])
        - np.maximum(boxes[:, 1], other[1]),
        0.0,
        None,
        )
    overlap = overlap_w * overlap_h
    hit = overlap > 0
    own_area = np.maximum((boxes[:, 2] - boxes[:, 0]) *
                            (boxes[:, 3] - boxes[:, 1]),
                            1.0)
    other_area = max((other[2] - other[0]) * (other[3] - other[1]), 1.0)
    costs = collision_weight * hit * (
        1.0 + overlap / np.minimum(own_area, other_area)
        )
    if safe_distance_px > 0:
        distances = _distance_bbox_borders_to_bbox(boxes, other)
        near = np.clip(safe_distance_px - distances, 0.0, None)
        costs += proximity_weight * (near / safe_distance_px)**2
    return costs


def _global_candidate_search(
    unary_costs,
    candidate_boxes,
    *,
    collision_weight,
    proximity_weight,
    safe_distance_px,
    n_restarts,
    max_iter,
    random_state,
    ) -> np.ndarray:
    n = len(unary_costs)
    if n <= 1:
        return np.asarray([int(np.argmin(unary_costs[0]))], dtype=int)
    rng = np.random.default_rng(random_state)

    def pair_vector(i, boxes_i, chosen):
        result = np.zeros(len(boxes_i), dtype=float)
        for j in range(n):
            if j != i and chosen[j] >= 0:
                result += _pair_cost_vector(
                    boxes_i,
                    candidate_boxes[j][chosen[j]],
                    collision_weight=collision_weight,
                    proximity_weight=proximity_weight,
                    safe_distance_px=safe_distance_px,
                    )
        return result

    def improve(chosen):
        for _ in range(max(1, int(max_iter))):
            changed = False
            for i in rng.permutation(n):
                best = int(
                    np.argmin(
                        unary_costs[i]
                        + pair_vector(i, candidate_boxes[i], chosen)
                        )
                    )
                if best != chosen[i]:
                    chosen[i] = best
                    changed = True
            if not changed:
                break
        return chosen

    def total_cost(chosen):
        total = sum(unary_costs[i][chosen[i]] for i in range(n))
        for i in range(n):
            for j in range(i):
                total += _pair_cost_vector(
                    candidate_boxes[i][chosen[i]:chosen[i] + 1],
                    candidate_boxes[j][chosen[j]],
                    collision_weight=collision_weight,
                    proximity_weight=proximity_weight,
                    safe_distance_px=safe_distance_px,
                    )[0]
        return float(total)

    starts = [
        np.asarray([np.argmin(cost) for cost in unary_costs], dtype=int)
        ]
    for _ in range(max(0, int(n_restarts) - 1)):
        chosen = np.full(n, -1, dtype=int)
        for i in rng.permutation(n):
            chosen[i] = int(
                np.argmin(
                    unary_costs[i]
                    + pair_vector(i, candidate_boxes[i], chosen)
                    )
                )
        starts.append(chosen)

    solutions = [improve(chosen.copy()) for chosen in starts]
    return min(solutions, key=total_cost)


def find_annotate_positions(
    fig,
    ax,
    x,
    y,
    t,
    *,
    fontdict=None,
    text_kwargs: Mapping | None = None,
    avoid_points=None,
    avoid_paths=None,
    avoid_transform=None,
    avoid_artists=None,
    allow_artists=None,
    auto_avoid: bool | None = None,
    mode: str = "mono_fast",
    fast_fallback: str = "exact",
    radii_px: Sequence[float] | None = None,
    n_angles: int = 24,
    safe_distance_px: float = 8.0,
    ignore_anchor_px: float = 5.0,
    path_sample_step_px: float = 4.0,
    collision_weight: float = 1000.0,
    proximity_weight: float = 20.0,
    distance_weight: float = 1.0,
    boundary_weight: float = 2000.0,
    preferred_direction=(1.0, 1.0),
    direction_weight: float = 0.15,
    annotation_collision_weight: float = 5000.0,
    annotation_proximity_weight: float = 40.0,
    annotation_safe_distance_px: float = 6.0,
    n_restarts: int = 6,
    max_iter: int = 30,
    random_state: int | None = 0,
    tabsize: int = 4,
    verify_metric: bool = True,
    ) -> np.ndarray:
    """
    Jointly place one or more labels and return an ``(N, 2)`` array.

    Each label receives a discrete candidate set consisting of its anchor and
    polar rings controlled by ``radii_px`` and ``n_angles``.  Unary costs cover
    obstacles, the shortest text-border-to-anchor distance, axes boundaries
    and direction preference.  Obstacle and pairwise spacing likewise use
    shortest border distances, while actual intersections are penalized by
    separate collision terms.  The joint solver uses independent and
    randomized greedy initial assignments followed by discrete coordinate
    descent (``n_restarts`` and ``max_iter``).
    """
    ax.viewLim.bounds  # 刷新自动缩放，确保坐标变换正确
    anchors, texts = _normalize_text_batch(x, y, t)
    if not len(anchors):
        return np.empty((0, 2), dtype=float)
    item_kwargs = _split_text_kwargs(text_kwargs, len(anchors))
    item_fontdicts = _split_fontdict(fontdict, len(anchors))
    _apply_default_alignments(item_kwargs, item_fontdicts)

    relatives = []
    transforms = []
    anchors_px = []
    for anchor, text, font, kwargs in zip(
            anchors, texts, item_fontdicts, item_kwargs):
        relative, transform, anchor_px = _measure_relative_bbox(
            fig, ax, anchor, text, font, kwargs, mode=mode,
            fast_fallback=fast_fallback, tabsize=tabsize,
            verify_metric=verify_metric,
            )
        relatives.append(relative)
        transforms.append(transform)
        anchors_px.append(anchor_px)
    anchors_px = np.asarray(anchors_px, dtype=float)

    # Probes are removed before existing artists are collected, so temporary
    # measurement text never becomes an obstacle.
    geometry = _collect_obstacle_geometry(
        fig,
        ax,
        avoid_points=avoid_points,
        avoid_paths=avoid_paths,
        avoid_transform=avoid_transform,
        avoid_artists=avoid_artists,
        allow_artists=allow_artists,
        auto_avoid=auto_avoid,
        path_sample_step_px=path_sample_step_px,
        )

    preferred = np.asarray(preferred_direction, dtype=float)
    preferred_norm = np.linalg.norm(preferred)
    if preferred_norm:
        preferred = preferred / preferred_norm

    axes_bb = ax.bbox
    positions = []
    boxes = []
    unary = []
    for anchor_px, relative in zip(anchors_px, relatives):
        candidates, candidate_boxes, diagonal = _candidate_positions(
            anchor_px, relative, radii_px, n_angles
            )
        costs = _obstacle_costs(
            candidate_boxes,
            anchor_px,
            geometry,
            ignore_anchor_px=ignore_anchor_px,
            safe_distance_px=safe_distance_px,
            collision_weight=collision_weight,
            proximity_weight=proximity_weight,
            )
        displacement = candidates - anchor_px
        center_distances = np.linalg.norm(displacement, axis=1)
        anchor_border_distances = _distance_point_to_bbox_borders(
            anchor_px, candidate_boxes
            )
        costs += (
            distance_weight * anchor_border_distances
            / max(diagonal, 20.0)
            )

        overflow = (
            np.clip(axes_bb.x0 - candidate_boxes[:, 0], 0.0, None)
            + np.clip(candidate_boxes[:, 2] - axes_bb.x1, 0.0, None)
            + np.clip(axes_bb.y0 - candidate_boxes[:, 1], 0.0, None)
            + np.clip(candidate_boxes[:, 3] - axes_bb.y1, 0.0, None)
            )
        costs += boundary_weight * (overflow > 0) * (1.0 + overflow/10.0)
        if preferred_norm:
            nonzero = center_distances > 0
            directions = np.zeros_like(displacement)
            directions[nonzero] = (
                displacement[nonzero] / center_distances[nonzero, None]
                )
            costs -= direction_weight * (directions@preferred)
        positions.append(candidates)
        boxes.append(candidate_boxes)
        unary.append(costs)

    chosen = _global_candidate_search(
        unary,
        boxes,
        collision_weight=annotation_collision_weight,
        proximity_weight=annotation_proximity_weight,
        safe_distance_px=annotation_safe_distance_px,
        n_restarts=n_restarts,
        max_iter=max_iter,
        random_state=random_state,
        )
    return np.asarray([
        transform.inverted().transform(positions[i][chosen[i]])
        for i, transform in enumerate(transforms)
        ],
                        dtype=float)


def find_annotate_position(
    fig,
    ax,
    anchor,
    text: str,
    **kwargs,
    ) -> np.ndarray:
    """Backward-compatible single-label wrapper around the global solver."""
    anchor = np.asarray(anchor, dtype=float)
    if anchor.shape != (2, ):
        raise ValueError("anchor must have shape (2,)")
    return find_annotate_positions(
        fig, ax, anchor[0], anchor[1], text, **kwargs
        )[0]


# Historical typo retained as a harmless alias.
find_anotate_position = find_annotate_position


def text_artist_uses_complex_layout(
        text_artist, text_kwargs: dict
    ) -> bool:
    """
    Return True when ``mono_fast`` should defer to exact measurement.

    The fast estimator intentionally targets ordinary monospace labels, not
    TeX/MathText or strongly shape-changing bbox styles.
    """
    text = str(text_artist.get_text())

    # 3.11 multiline spacing follows per-line font metrics. Use the native
    # renderer here instead of approximating the new layout semantics.
    if '\n' in text and hasattr(text_artist, 'get_linespacing'):
        return True

    if bool(text_artist.get_usetex()):
        return True

    # A minimal MathText check. Escaped dollar signs are uncommon for labels;
    # users can force mode="exact" when needed.
    if "$" in text:
        return True

    bbox_kwargs = text_kwargs.get("bbox")
    if bbox_kwargs:
        boxstyle = bbox_kwargs.get("boxstyle", "square,pad=0.3")
        name = _parse_boxstyle_name(boxstyle)

        if name is not None and name not in _FAST_SUPPORTED_BOXSTYLES:
            return True

        # A BoxStyle instance rather than a simple string is safest in exact mode.
        if name is None and boxstyle is not None:
            return True

    return False


# ---------------------------------------------------------------------------
# Public ax.text-like wrapper
# ---------------------------------------------------------------------------


def text_better(
    fig,
    ax,
    x,
    y,
    t=None,
    fontdict=None,
    *,
    avoid_points=None,
    avoid_paths=None,
    avoid_transform=None,
    avoid_artists=None,
    allow_artists=None,
    auto_avoid: bool | None = None,
    placement_mode: Literal["mono_fast", "exact"] = "mono_fast",
    placement_kwargs: dict | None = None,
    **kwargs,
    ):
    """
    ``ax.text``-like automatic placement for one label or a broadcast batch.

    ``x``, ``y`` and ``t`` accept scalars or broadcastable arrays.  Common text
    properties such as ``color``, ``fontsize``, ``fontweight``, ``rotation``,
    ``ha``, ``va`` and ``bbox`` may also be length-N lists or ndarrays.  A
    single call is optimized jointly, including label-to-label overlap.

    The old keyword ``s=`` remains an alias for ``t=``.  Alignment defaults to
    ``ha='center', va='center'``, making the text-box center its anchor.

    When neither manual points nor paths are supplied, visible lines,
    collections, patches and text already on the axes are avoided by default.
    ``avoid_artists`` and ``allow_artists`` accept the selectors documented by
    :func:`select_artists`.  Images are not in the default obstacle set.

    Examples
    --------
    >>> text_better(
    ...     fig, ax, [1, 2], [3, 3.1], ["轨迹 A", "轨迹 B"],
    ...     fontsize=[10, 12], color=["tab:blue", "tab:red"],
    ...     avoid_artists={"label": r"^trajectory-"},
    ...     bbox=dict(boxstyle="round,pad=0.25", fc="white"),
    ... )

    Useful ``placement_kwargs`` recipes include::

        # Stay close while retaining collision avoidance.
        dict(radii_px=(6, 12, 22, 36, 56), distance_weight=4.0,
             direction_weight=0.0, annotation_safe_distance_px=4.0)

        # Prefer generous whitespace in a less crowded figure.
        dict(radii_px=(20, 40, 70, 100, 140), distance_weight=0.6,
             safe_distance_px=12.0, annotation_safe_distance_px=10.0)

        # Faster approximate layout for many labels.
        dict(radii_px=(12, 28, 50), n_angles=12, n_restarts=2,
             max_iter=12, path_sample_step_px=7.0)

        # More exhaustive layout (combine with placement_mode="exact").
        dict(n_angles=48, n_restarts=12, max_iter=60,
             path_sample_step_px=2.0)

    Returns
    -------
    matplotlib.text.Text or list[matplotlib.text.Text]
        A native Text for scalar input, otherwise one Text per broadcast item.
    """
    legacy_s = kwargs.pop("s", None)
    if t is None and legacy_s is None:
        raise TypeError("text_better() is missing required argument: 't'")
    if t is not None and legacy_s is not None:
        raise TypeError(
            "pass text as either 't' or the legacy 's', not both"
            )
    text = legacy_s if t is None else t

    placement = dict(placement_kwargs or {})
    positions = find_annotate_positions(
        fig,
        ax,
        x,
        y,
        text,
        fontdict=fontdict,
        text_kwargs=kwargs,
        avoid_points=avoid_points,
        avoid_paths=avoid_paths,
        avoid_transform=avoid_transform,
        avoid_artists=avoid_artists,
        allow_artists=allow_artists,
        auto_avoid=auto_avoid,
        mode=placement_mode,
        **placement,
        )

    anchors, texts = _normalize_text_batch(x, y, text)
    item_kwargs = _split_text_kwargs(kwargs, len(anchors))
    item_fontdicts = _split_fontdict(fontdict, len(anchors))
    _apply_default_alignments(item_kwargs, item_fontdicts)
    artists = [
        ax.text(
            pos[0],
            pos[1],
            item_text,
            fontdict=item_fontdict,
            **item_kwargs_i
            ) for pos, item_text, item_fontdict, item_kwargs_i in
        zip(positions, texts, item_fontdicts, item_kwargs)
        ]
    scalar_input = (
        _is_scalar_input(x) and _is_scalar_input(y)
        and _is_scalar_input(text)
        )
    return artists[0] if scalar_input else artists
