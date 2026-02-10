from __future__ import annotations
from typing import List, Tuple

from src.core.scripts.models import Point, PointType
from src.core.scripts.postprocessing.newick import build_newick  # adapte le chemin si besoin


def compute_newick(points: List[Point]) -> str:
    if not points:
        return "();"

    tips = [p for p in points if p.ptype == PointType.TIP]
    internals = [p for p in points if p.ptype in (PointType.NODE, PointType.ROOT)]
    corners = [p for p in points if p.ptype == PointType.CORNER]

    if not tips or not internals or not corners:
        return "();"

    # IMPORTANT: newick_clean trie leaves par y décroissant (haut->bas) :contentReference[oaicite:2]{index=2}
    tips_sorted = sorted(tips, key=lambda p: p.y, reverse=True)

    leaves_xy: List[Tuple[int, int]] = [(int(round(p.x)), int(round(p.y))) for p in tips_sorted]
    internals_xy: List[Tuple[int, int]] = [(int(round(p.x)), int(round(p.y))) for p in internals]
    corners_xy: List[Tuple[int, int]] = [(int(round(p.x)), int(round(p.y))) for p in corners]

    leaf_names = []
    for i, p in enumerate(tips_sorted, start=1):
        name = (p.label or "").strip()
        leaf_names.append(name if name else f"L{i}")

    newick = build_newick(
        leaves=leaves_xy,
        internals=internals_xy,
        corners=corners_xy,
        x_tol=18.0,
        y_tol=14.0,
        leaf_names=leaf_names,
        decimals=6
    )
    return newick if newick is not None else "();"
