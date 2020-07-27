import sys
import math
import projection
from movement_sphere import Movement
import json
import cfg

def proj(y, x):
	x = x - cfg.camerad["w"]/2
	y = y - cfg.camerad["h"]/2
	X = x / cfg.camerad["w"] * cfg.camerad["W"]
	Y = x / cfg.camerad["h"] * cfg.camerad["H"]
	lon = math.atan(X / cfg.camerad["F"])
	lat = math.atan(Y * math.cos(lon) / cfg.camerad["F"])
	return lat, lon

points1 = [
	{
		"x" : 100,
		"y" : 100
	},
	{
		"x" : cfg.camerad["w"] - 100,
		"y" : 100
	},
	{
		"x" : cfg.camerad["w"] - 100,
		"y" : cfg.camerad["h"] - 100
	},
	{
		"x" : 100,
		"y" : cfg.camerad["h"] - 100
	},
	{
		"x" : cfg.camerad["w"]/2,
		"y" : cfg.camerad["h"]/2
	}
]

for point in points1:
	lat, lon = proj(point["y"], point["x"])
	point["lat"] = lat
	point["lon"] = lon

print(points1)

