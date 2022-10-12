from enum import Enum, auto


class Mode(Enum):
    """Point cloud Folder to choose from."""

    TEST = auto()
    RAW = auto()


class PCFormats(Enum):
    """Readable point cloud formats for Open3D"""

    XYZ = auto()
    XYZN = auto()
    XYZRGBA = auto()
    PTS = auto()
    PLY = auto()
    PCD = auto()
