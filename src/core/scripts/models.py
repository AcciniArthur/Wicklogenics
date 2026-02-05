from dataclasses import dataclass
from enum import Enum

class PointType(str, Enum):
    ROOT = "root"
    NODE = "node"
    CORNER = "corner"
    TIP = "tip"

POINT_COLORS = {
    PointType.ROOT: "green",
    PointType.NODE: "red",
    PointType.CORNER: "yellow",
    PointType.TIP: "blue",
}

@dataclass
class Point:
    x: float  # image coords (pixels)
    y: float  # image coords (pixels)
    ptype: PointType
    label: str | None = None
