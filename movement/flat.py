import common
import math
import json
import numpy as np

class Movement(object):

	def apply(self, positions, proj):
		npositions = []
		for y, x in positions:
			nx = x * math.cos(self.a) - y * math.sin(self.a) + self.dx
			ny = y * math.cos(self.a) + x * math.sin(self.a) + self.dy		
			npositions.append((ny, nx))

		return npositions

	def reverse(self, positions, proj):
		npositions = []
		for y, x in positions:
			nx = (x-self.dx) * math.cos(self.a) + (y-self.dy) * math.sin(self.a)
			ny = (y-self.dy) * math.cos(self.a) - (x-self.dx) * math.sin(self.a)
			npositions.append((ny, nx))

		return npositions

	def magnitude(self):
		return self.dx**2 + self.dy**2 + self.a**2

	# move pi1, pi2 to p1, p2
	def __init__(self, angle, dy, dx):
		self.dx = dx
		self.dy = dy
		self.a = angle

	def serialize(self):
		return json.dumps({"dy" : self.dy, "dx" : self.dx, "angle" : self.a*180/math.pi})

	@staticmethod
	def deserialize(ser):
		s = json.loads(ser)
		return Movement(s["angle"]*math.pi/180, s["dy"], s["dx"])

	@staticmethod
	def build(pi1, pi2, p1, p2, cd, debug=False):
		diy, dix = common.norm((pi2[0] - pi1[0], pi2[1] - pi1[1]))
		dy, dx = common.norm((p2[0] - p1[0], p2[1] - p1[1]))

		cosa = diy*dy + dix*dx
		sina = dix*dy - diy*dx

		if cosa > 1:
			cosa = 1
		if cosa < -1:
			cosa = -1

		a = math.asin(sina)

		if cosa < 0:
			a = math.pi - a

		dx = 0
		dy = 0
		t = Movement(a, dy, dx)
		ty, tx = t.apply([(pi1[0], pi1[1])])[0]
		dy = p1[0] - ty
		dx = p1[1] - tx
		return Movement(a, dy, dx)

	@staticmethod
	def average(ts):
		angles = []
		dxs = []
		dys = []
		for t in ts:
			angles.append(t.a)
			dxs.append(t.dx)
			dys.append(t.dy)
		angle = np.average(angles)
		dy = np.average(dys)
		dx = np.average(dxs)
		t = Movement(angle, dy, dx)
		dangle = np.std(angles)
		ddy = np.std(dys)
		ddx = np.std(dxs)
		return t

