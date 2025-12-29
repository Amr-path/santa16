"""
geometry.py - Core geometry for Santa 2025
Uses EXACT tree polygon from official Kaggle starter code.
"""
import math
import numpy as np
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union
from typing import List, Tuple, Optional

# High precision settings (from official code)
getcontext().prec = 25
SCALE_FACTOR = Decimal('1e15')

# Tree dimensions (from official code)
TRUNK_W = Decimal('0.15')
TRUNK_H = Decimal('0.2')
BASE_W = Decimal('0.7')
MID_W = Decimal('0.4')
TOP_W = Decimal('0.25')
TIP_Y = Decimal('0.8')
TIER_1_Y = Decimal('0.5')
TIER_2_Y = Decimal('0.25')
BASE_Y = Decimal('0.0')
TRUNK_BOTTOM_Y = -TRUNK_H

# Build the EXACT polygon from official code (15 vertices)
def _build_base_polygon():
    """Build base tree polygon using official coordinates."""
    coords = [
        # Start at Tip
        (float(Decimal('0.0') * SCALE_FACTOR), float(TIP_Y * SCALE_FACTOR)),
        # Right side - Top Tier
        (float(TOP_W / Decimal('2') * SCALE_FACTOR), float(TIER_1_Y * SCALE_FACTOR)),
        (float(TOP_W / Decimal('4') * SCALE_FACTOR), float(TIER_1_Y * SCALE_FACTOR)),
        # Right side - Middle Tier
        (float(MID_W / Decimal('2') * SCALE_FACTOR), float(TIER_2_Y * SCALE_FACTOR)),
        (float(MID_W / Decimal('4') * SCALE_FACTOR), float(TIER_2_Y * SCALE_FACTOR)),
        # Right side - Bottom Tier
        (float(BASE_W / Decimal('2') * SCALE_FACTOR), float(BASE_Y * SCALE_FACTOR)),
        # Right Trunk
        (float(TRUNK_W / Decimal('2') * SCALE_FACTOR), float(BASE_Y * SCALE_FACTOR)),
        (float(TRUNK_W / Decimal('2') * SCALE_FACTOR), float(TRUNK_BOTTOM_Y * SCALE_FACTOR)),
        # Left Trunk
        (float(-(TRUNK_W / Decimal('2')) * SCALE_FACTOR), float(TRUNK_BOTTOM_Y * SCALE_FACTOR)),
        (float(-(TRUNK_W / Decimal('2')) * SCALE_FACTOR), float(BASE_Y * SCALE_FACTOR)),
        # Left side - Bottom Tier
        (float(-(BASE_W / Decimal('2')) * SCALE_FACTOR), float(BASE_Y * SCALE_FACTOR)),
        # Left side - Middle Tier
        (float(-(MID_W / Decimal('4')) * SCALE_FACTOR), float(TIER_2_Y * SCALE_FACTOR)),
        (float(-(MID_W / Decimal('2')) * SCALE_FACTOR), float(TIER_2_Y * SCALE_FACTOR)),
        # Left side - Top Tier
        (float(-(TOP_W / Decimal('4')) * SCALE_FACTOR), float(TIER_1_Y * SCALE_FACTOR)),
        (float(-(TOP_W / Decimal('2')) * SCALE_FACTOR), float(TIER_1_Y * SCALE_FACTOR)),
    ]
    return Polygon(coords)

BASE_POLYGON = _build_base_polygon()

# Simple coordinates for fast operations (unscaled)
TREE_COORDS_SIMPLE = [
    (0.0, 0.8),        # tip
    (0.125, 0.5),      # top tier outer right
    (0.0625, 0.5),     # top tier inner right
    (0.2, 0.25),       # middle tier outer right
    (0.1, 0.25),       # middle tier inner right
    (0.35, 0.0),       # bottom tier outer right
    (0.075, 0.0),      # trunk top right
    (0.075, -0.2),     # trunk bottom right
    (-0.075, -0.2),    # trunk bottom left
    (-0.075, 0.0),     # trunk top left
    (-0.35, 0.0),      # bottom tier outer left
    (-0.1, 0.25),      # middle tier inner left
    (-0.2, 0.25),      # middle tier outer left
    (-0.0625, 0.5),    # top tier inner left
    (-0.125, 0.5),     # top tier outer left
]
SIMPLE_BASE_POLYGON = Polygon(TREE_COORDS_SIMPLE)

# Tree dimensions
TREE_WIDTH = 0.7
TREE_HEIGHT = 1.0


def make_tree_polygon_scaled(center_x: Decimal, center_y: Decimal, angle: Decimal) -> Polygon:
    """Create tree polygon using high-precision scaled coordinates (like official code)."""
    rotated = affinity.rotate(BASE_POLYGON, float(angle), origin=(0, 0))
    translated = affinity.translate(
        rotated,
        xoff=float(center_x * SCALE_FACTOR),
        yoff=float(center_y * SCALE_FACTOR)
    )
    return translated


def make_tree_polygon(x: float, y: float, angle_deg: float) -> Polygon:
    """Create tree polygon with simple float coordinates (faster)."""
    poly = SIMPLE_BASE_POLYGON
    if angle_deg != 0:
        poly = affinity.rotate(poly, angle_deg, origin=(0, 0))
    if x != 0 or y != 0:
        poly = affinity.translate(poly, xoff=x, yoff=y)
    return poly


# Alias
transform_tree = make_tree_polygon


def has_collision(tree_poly: Polygon, other_polys: List[Polygon]) -> bool:
    """Check if tree_poly overlaps any other polygon (touching OK)."""
    for poly in other_polys:
        if tree_poly.intersects(poly) and not tree_poly.touches(poly):
            return True
    return False


def has_collision_strtree(tree_poly: Polygon, tree_index: STRtree, all_polys: List[Polygon]) -> bool:
    """Fast collision check using STRtree spatial index."""
    candidates = tree_index.query(tree_poly)
    for idx in candidates:
        if tree_poly.intersects(all_polys[idx]) and not tree_poly.touches(all_polys[idx]):
            return True
    return False


def compute_bounds(polygons: List[Polygon]) -> Tuple[float, float, float, float]:
    """Compute bounding box of all polygons."""
    if not polygons:
        return (0, 0, 0, 0)
    union = unary_union(polygons)
    return union.bounds


def compute_bounding_square_side(placements: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side from placements."""
    if not placements:
        return 0.0
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return max(width, height)


def bounding_square_side_from_polys(polygons: List[Polygon]) -> float:
    """Compute bounding square side from polygon list."""
    if not polygons:
        return 0.0
    bounds = unary_union(polygons).bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return max(width, height)


def center_placements(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Center placements around origin."""
    if not placements:
        return placements
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]


def normalize_to_origin(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Shift placements so min x,y are at 0."""
    if not placements:
        return placements
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    return [(x - bounds[0], y - bounds[1], d) for x, y, d in placements]


def check_all_overlaps(placements: List[Tuple[float, float, float]]) -> List[Tuple[int, int]]:
    """Find all overlapping pairs."""
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    overlaps = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                overlaps.append((i, j))
    return overlaps
