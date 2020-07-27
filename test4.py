import sys
import math
import projection
from movement_sphere import Movement
import json
import cfg
import numpy as np

ser = "{\"rot\": [-2.8727688157243295e-05, -2.694664812967076e-05, -0.01713579184483744, 0.999853170763812], \"h\": 711, \"w\": 1101, \"H\": 15.4, \"W\": 23.1, \"F\": 35}"
t = Movement.deserialize(ser)
rv = t.rot.as_rotvec()
l = np.dot(rv,rv)**0.5
print(l*180/math.pi)
sys.exit()

y1_128 = 631
x1_128 = 284

y1_130 = 621
x1_130 = 375

npos = t.reverse([(y1_130, x1_130)])
ny = npos[0][0]
nx = npos[0][1]
print("-------\n130 : ", y1_130, x1_130, "\n130': ", ny, nx, "\n128 : ", y1_128, x1_128)

y2_128 = 301
x2_128 = 4202

y2_130 = 274
x2_130 = 4292

npos = t.reverse([(y2_130, x2_130)])
ny = npos[0][0]
nx = npos[0][1]
print("-------\n130 : ", y2_130, x2_130, "\n130': ", ny, nx, "\n128 : ", y2_128, x2_128)

y3_128 = 1913
x3_128 = 2343

y3_130 = 1894
x3_130 = 2435

npos = t.reverse([(y3_130, x3_130)])
ny = npos[0][0]
nx = npos[0][1]
print("-------\n130 : ", y3_130, x3_130, "\n130': ", ny, nx, "\n128 : ", y3_128, x3_128)


