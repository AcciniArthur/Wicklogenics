from src.core.scripts.models import Point, PointType
from src.core.scripts.v0 import v0

_model = None

def run_pipeline(image_path: str) -> list[Point]:
    global _model
    if _model is None:
        _model = v0()
        _model.load(device="cpu")

    data = _model.convert(image_path, device="cpu")

    points: list[Point] = []

    # nodes
    for t in data.get("nodes", []):
        x, y = float(t[0]), float(t[1])
        points.append(Point(x, y, PointType.NODE, None))

    # corners
    for t in data.get("corners", []):
        x, y = float(t[0]), float(t[1])
        points.append(Point(x, y, PointType.CORNER, None))

    # leaves -> tips + label
    leaf_names = data.get("leaf_names", [])
    leaves = data.get("leaves", [])
    for i, t in enumerate(leaves):
        x, y = float(t[0]), float(t[1])
        label = leaf_names[i] if i < len(leaf_names) else f"tip{i+1}"
        points.append(Point(x, y, PointType.TIP, label))

    return points
