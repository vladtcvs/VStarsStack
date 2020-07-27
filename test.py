import sys
import math
import projection
from movement_sphere import Movement
import json
import cfg

star1_1 = {
            "y": 1582.0,
            "x": 2652.300048828125,
            "lat": 0.0056894634822371734,
            "lon": 0.04699855890939322,
            "size": 3.448263645172119
        }

star1_2 = {
            "y": 1583.8333740234375,
            "x": 2654.833251953125,
            "lat": 0.005950129217935465,
            "lon": 0.047358472982173765,
            "size": 3.3749051094055176
        }



star2_1 = {
            "y": 2059.5,
            "x": 3560.0,
            "lat": 0.0724436064908385,
            "lon": 0.17449265068698425,
            "size": 3.6401548385620117
        }

star2_2 = {
            "y": 2061.5,
            "x": 3562.5,
            "lat": 0.07271818293904121,
            "lon": 0.17483788859734675,
            "size": 3.5356338024139404
        }

pi1 = (star1_1["lat"], star1_1["lon"])
p1 = (star1_2["lat"], star1_2["lon"])
pi2 = (star2_1["lat"], star2_1["lon"])
p2 = (star2_2["lat"], star2_2["lon"])

#pi1 = (star1_1["y"], star1_1["x"])
#p1 = (star1_2["y"], star1_2["x"])
#pi2 = (star2_1["y"], star2_1["x"])
#p2 = (star2_2["y"], star2_2["x"])

cd = cfg.camerad
t = Movement.build(pi1, pi2, p1, p2, cd)

y = star1_1["y"]
x = star1_1["x"]
cy = star1_2["y"]
cx = star1_2["x"]
ny, nx = t.apply(y, x)
print(y, x, "/", ny, nx, "/", cy, cx)

lat = star1_1["lat"]
lon = star1_1["lon"]

proj = projection.Projection(cd["W"], cd["H"], cd["F"], cd["w"], cd["h"])
nlat, nlon = proj.project(ny, nx)

clat = star1_2["lat"]
clon = star1_2["lon"]
print(lat, lon, "/", nlat, nlon, "/", clat, clon)

print(t.serialize())

#y = star2_1["y"]
#x = star2_1["x"]
#cy = star2_2["y"]
#cx = star2_2["x"]
#ny, nx = t.apply(y, x)
#print(y, x, "/", ny, nx, "/", cy, cx)

