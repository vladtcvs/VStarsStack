from enum import Enum

class ProjectionType(Enum):
    NoneProjection = 0
    Perspective = 1
    Orthographic = 2
    Equirectangular = 3
