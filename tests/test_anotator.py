import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from mtp_initializer.annotator import (
    _distance_bbox_borders_to_bbox,
    _distance_point_to_bbox_borders,
    _distance_points_to_bbox_border,
    find_annotate_position,
    select_artists,
    text_better,
)


def _intersection_area(left, right):
    width = max(0.0, min(left.x1, right.x1)-max(left.x0, right.x0))
    height = max(0.0, min(left.y1, right.y1)-max(left.y0, right.y0))
    return width*height


def test_scalar_call_is_centered_and_legacy_s_is_supported():
    fig, ax = plt.subplots()
    artist = text_better(
        fig, ax, 0.5, 0.5, s="legacy", avoid_artists=False,
        placement_kwargs={"preferred_direction": (0, 0)},
        )

    assert isinstance(artist, Text)
    assert artist.get_horizontalalignment() == "center"
    assert artist.get_verticalalignment() == "center"
    assert np.isfinite(artist.get_position()).all()
    plt.close(fig)


def test_batch_styles_are_broadcast_and_labels_are_jointly_separated():
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    artists = text_better(
        fig, ax,
        [0.5, 0.5, 0.5],
        0.5,
        ["alpha", "beta-long", "gamma"],
        avoid_artists=False,
        placement_mode="exact",
        fontsize=np.array([9, 12, 15]),
        color=np.array(["red", "green", "blue"]),
        bbox=[
            {"facecolor": "white", "pad": 0.2},
            {"facecolor": "ivory", "pad": 0.2},
            {"facecolor": "aliceblue", "pad": 0.2},
            ],
        placement_kwargs={
            "radii_px": (24, 48, 72),
            "preferred_direction": (0, 0),
            "n_restarts": 8,
            },
        )

    assert isinstance(artists, list)
    assert [artist.get_fontsize() for artist in artists] == [9, 12, 15]
    assert [artist.get_color() for artist in artists] == ["red", "green", "blue"]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_bbox_patch().get_window_extent(renderer) for artist in artists]
    assert all(
        _intersection_area(boxes[i], boxes[j]) == 0
        for i in range(len(boxes)) for j in range(i)
        )
    plt.close(fig)


def test_single_rgb_tuple_is_not_mistaken_for_three_item_batch():
    fig, ax = plt.subplots()
    artists = text_better(
        fig, ax, [0.2, 0.5, 0.8], 0.5, ["a", "b", "c"],
        color=(0.1, 0.2, 0.3), avoid_artists=False,
        )
    assert [artist.get_color() for artist in artists] == [(0.1, 0.2, 0.3)]*3
    plt.close(fig)


def test_artist_selectors_support_regex_class_names_and_mixed_type_batches():
    fig, ax = plt.subplots()
    signal, = ax.plot([0, 1], [0, 1], label="signal-main")
    signal_aux, = ax.plot([0, 1], [0.2, 0.8], label="signal-aux")
    background, = ax.plot([0, 1], [1, 0], label="background")
    patch = Rectangle((0.1, 0.1), 0.2, 0.2, label="region")
    ax.add_patch(patch)
    image = ax.imshow(np.ones((2, 2)), label="background-image")

    assert select_artists(ax, r"^signal-main$") == [signal]
    assert select_artists(ax, {"label": r"^signal-(main|aux)$"}) == [
        signal, signal_aux,
        ]
    assert signal in select_artists(
        ax, {"type": "Line2D", "label": r"^signal-"}
        )
    assert signal in select_artists(
        ax, {"type": "matplotlib.lines.Line2D", "label": r"main$"}
        )

    mixed_types = select_artists(ax, ["line", "Rectangle"])
    assert signal in mixed_types and signal_aux in mixed_types
    assert patch in mixed_types
    mapping_types = select_artists(
        ax, {"type": ["Line2D", Rectangle], "label": r"^(signal|region)"}
        )
    assert signal in mapping_types and patch in mapping_types
    assert background not in mapping_types
    assert select_artists(ax, "image") == [image]

    union = select_artists(ax, [signal, {"label": "region"}])
    assert signal in union and patch in union and background not in union
    assert not any(isinstance(item, matplotlib.image.AxesImage)
                   for item in select_artists(ax))
    plt.close(fig)


def test_artist_whitelist_removes_selected_obstacle():
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    line, = ax.plot([0, 1], [0.5, 0.5], label="blocking")

    common = dict(
        placement_mode="exact",
        avoid_artists=line,
        placement_kwargs={
            "preferred_direction": (0, 0),
            "ignore_anchor_px": 0,
            "radii_px": (30, 60),
            },
        )
    baseline = text_better(
        fig, ax, 0.5, 0.5, "label", placement_mode="exact",
        avoid_artists=False, placement_kwargs=common["placement_kwargs"],
        )
    baseline_position = baseline.get_position()
    baseline.remove()

    blocked = text_better(fig, ax, 0.5, 0.5, "label", **common)
    assert not np.allclose(blocked.get_position(), baseline_position)
    blocked.remove()

    allowed = text_better(
        fig, ax, 0.5, 0.5, "label", allow_artists=line, **common
        )
    np.testing.assert_allclose(allowed.get_position(), baseline_position)
    plt.close(fig)


def test_annotation_arrow_patch_is_selectable_and_avoided_before_draw():
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    annotation = ax.annotate(
        "", (0.8, 0.5), xytext=(0.2, 0.5),
        arrowprops={"arrowstyle": "->"},
        )

    assert select_artists(ax, "FancyArrowPatch") == [annotation.arrow_patch]
    assert select_artists(ax, "arrow") == [annotation.arrow_patch]
    assert annotation.arrow_patch in select_artists(ax)

    placement = {
        "preferred_direction": (0, 0),
        "ignore_anchor_px": 0,
        "radii_px": (30, 60),
        }
    baseline = text_better(
        fig, ax, 0.5, 0.5, "label", placement_mode="exact",
        avoid_artists=False, placement_kwargs=placement,
        )
    baseline_position = baseline.get_position()
    baseline.remove()

    blocked = text_better(
        fig, ax, 0.5, 0.5, "label", placement_mode="exact",
        avoid_artists="FancyArrowPatch", placement_kwargs=placement,
        )
    assert not np.allclose(blocked.get_position(), baseline_position)
    blocked.remove()

    allowed = text_better(
        fig, ax, 0.5, 0.5, "label", placement_mode="exact",
        avoid_artists="arrow", allow_artists=annotation,
        placement_kwargs=placement,
        )
    np.testing.assert_allclose(allowed.get_position(), baseline_position)
    plt.close(fig)


def test_default_artist_avoidance_is_disabled_by_manual_geometry():
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    ax.plot([0, 1], [0.5, 0.5])
    placement = {
        "preferred_direction": (0, 0),
        "ignore_anchor_px": 0,
        "radii_px": (30, 60),
        }
    baseline = text_better(
        fig, ax, 0.5, 0.5, "label", placement_mode="exact",
        avoid_artists=False, placement_kwargs=placement,
        )
    baseline_position = baseline.get_position()
    baseline.remove()

    automatic = text_better(
        fig, ax, 0.5, 0.5, "label", placement_mode="exact",
        placement_kwargs=placement,
        )
    assert not np.allclose(automatic.get_position(), baseline_position)
    automatic.remove()

    manual = text_better(
        fig, ax, 0.5, 0.5, "label", placement_mode="exact",
        avoid_points=[[10, 10]], placement_kwargs=placement,
        )
    np.testing.assert_allclose(manual.get_position(), baseline_position)
    plt.close(fig)


def test_old_position_api_still_returns_one_xy_pair():
    fig, ax = plt.subplots()
    position = find_annotate_position(
        fig, ax, (0.3, 0.7), "one", avoid_artists=False,
        preferred_direction=(0, 0),
        )
    assert position.shape == (2,)
    assert np.isfinite(position).all()
    plt.close(fig)


def test_border_distances_handle_inside_overlap_containment_and_separation():
    bbox = np.array([0.0, 0.0, 10.0, 6.0])
    points = np.array([
        [5.0, 3.0],   # Inside: three pixels from the nearest border.
        [15.0, 3.0],  # Outside: five pixels from the right border.
        [0.0, 2.0],   # On the border.
        ])
    np.testing.assert_allclose(
        _distance_points_to_bbox_border(points, bbox), [3.0, 5.0, 0.0]
        )

    candidate_boxes = np.array([
        [0.0, 0.0, 10.0, 6.0],
        [10.0, 2.0, 14.0, 4.0],
        ])
    np.testing.assert_allclose(
        _distance_point_to_bbox_borders([5.0, 3.0], candidate_boxes),
        [3.0, 5.0],
        )

    other_boxes = np.array([
        [2.0, 2.0, 8.0, 4.0],    # Strictly contained: border gap two.
        [8.0, 2.0, 12.0, 4.0],   # Boundaries intersect.
        [12.0, 8.0, 14.0, 10.0], # Diagonally separated by (2, 2).
        ])
    np.testing.assert_allclose(
        _distance_bbox_borders_to_bbox(other_boxes, bbox),
        [2.0, 0.0, np.sqrt(8.0)],
        )
