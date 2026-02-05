from src.core.scripts.models import Point, PointType

def run_pipeline(image_path: str) -> list[Point]:
    """
    Remplace ceci par ta pipeline réelle (image -> points).
    Doit retourner des coords en pixels (coordonnées image).
    """
    return [
        Point(120, 120, PointType.ROOT, "root"),
        Point(220, 160, PointType.NODE),
        Point(260, 220, PointType.CORNER),
        Point(360, 260, PointType.TIP, "A"),
        Point(420, 300, PointType.TIP, "B"),
    ]
