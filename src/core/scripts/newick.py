from src.core.scripts.models import Point, PointType

def compute_newick(points: list[Point]) -> str:
    """
    Remplace par ton algo Newick basé sur les points.
    """
    tips = [p.label for p in points if p.ptype == PointType.TIP and p.label]
    if not tips:
        tips = []
    if len(tips) == 1:
        return f"({tips[0]});"
    if len(tips) == 2:
        return f"({tips[0]},{tips[1]});"
    return "(" + ",".join(tips) + ");"

