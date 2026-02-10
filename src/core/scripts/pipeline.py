from src.core.models import Point, PointType
from src.core.scripts.v0 import v0  # adapte le chemin

_model = None

def run_pipeline(image_path: str) -> list[Point]:
    global _model
    if _model is None:
        _model = v0()
        _model.load(device="cpu")

    data = _model.convert(image_path, device="cpu")

    points: list[Point] = []
    print("1111111111111111111111111111111111111111111")
    for t in data["nodes"]:
        x, y = t[0], t[1]
        points.append(Point(float(x), float(y), PointType.NODE, None))
    print("22222222222222222222222222222222222")
    for t in data["corners"]:
        x, y = t[0], t[1]
        points.append(Point(float(x), float(y), PointType.CORNER, None))
    print("3333333333333333333333333333333333333333")
    leaf_names = data.get("leaf_names", [])
    leaves = data.get("leaves", [])
    print("444444444444444444444444444444444444444444")
    for i, t in enumerate(leaves):
        x, y = t[0], t[1]
        label = leaf_names[i] if i < len(leaf_names) else f"tip{i+1}"
        points.append(Point(float(x), float(y), PointType.TIP, label))

    return points, label
