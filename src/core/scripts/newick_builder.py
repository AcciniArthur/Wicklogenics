from __future__ import annotations

from typing import List, Dict, Any

from src.core.models import Point as UiPoint, PointType
from src.core.scripts.postprocessing.newick import build_newick


def build_newick_from_ui_points(points: List[UiPoint]) -> str:
    """
    Convertit les points UI (pixels image) -> Newick string via newick.py
    - nodes: ROOT + NODE
    - corners: CORNER
    - tips: utilisés comme 'texts' (labels) pour nommer les feuilles
    """
    if not points:
        return "();"

    # Split
    roots = [p for p in points if p.ptype == PointType.ROOT]
    nodes = [p for p in points if p.ptype == PointType.NODE]
    corners = [p for p in points if p.ptype == PointType.CORNER]
    tips = [p for p in points if p.ptype == PointType.TIP]

    # nodes = root + nodes (l'algo cherche le min_x => root doit être dedans)
    nodes_in = [(p.x, p.y, 1.0) for p in (roots + nodes)]
    corners_in = [(p.x, p.y, 1.0) for p in corners]

    # x_leave doit correspondre au max x des points utilisés dans build_newick_from_points
    # Mais dans ton algo, x_leave est max(p.x) des points (nodes+corners).
    # Pour que get_nearest_label() colle aux TIP, on place les bbox "sur la ligne x_leave"
    x_leave = max([t[0] for t in nodes_in + corners_in]) if (nodes_in or corners_in) else 0.0

    # texts: on fabrique des "bbox" alignées sur x_leave, avec y_center = tip.y
    # pour que get_nearest_label(x_leave, node.y, ...) retourne le label du tip le plus proche en y.
    texts: List[Dict[str, Any]] = []
    for tip in tips:
        label = (tip.label or "leaf").strip() or "leaf"
        y = float(tip.y)
        texts.append({
            "text": label,
            "bbox": [float(x_leave), y - 1.0, float(x_leave), y + 1.0]
        })

    # Paramètres en pixels (à ajuster facilement)
    # margin = tolérance pour "alignement" des points (vertical/horizontal)
    # Si ton image est grande, augmente un peu.
    margin_px = 12.0
    max_distance_px = 40.0

    newick_obj = build_newick(
        nodes=nodes_in,
        corners=corners_in,
        scale=1.0,                # IMPORTANT: tes points UI sont déjà en pixels
        margin=margin_px,
        texts=texts,
        max_distance=max_distance_px
    )

    if newick_obj is None:
        return "();"

    return newick_obj.to_string()
