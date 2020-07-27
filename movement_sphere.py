import common
import math
import json
import numpy as np
from scipy.spatial.transform import Rotation

import projection

def p2vec(pos):
	lat = pos[0]
	lon = pos[1]
	return np.array([math.cos(lon)*math.cos(lat), math.sin(lon)*math.cos(lat), math.sin(lat)])

def vecmul(v1, v2):
	return np.cross(v1, v2)

def scmul(v1, v2):
	return np.dot(v1, v2)

def vecangle(v1, v2):
	d1 = np.linalg.norm(v1)
	d2 = np.linalg.norm(v2)
	return np.arccos(np.clip(np.dot(v1/d1, v2/d2), -1.0, 1.0))

class Movement(object):

	def apply(self, positions):
		ps = []
		for y, x in positions:
			lat, lon = self.proj.project(y, x)
			p = p2vec((lat, lon))
			ps.append(p)
		nps = self.rot.apply(ps)
		npositions = []
		for pp in nps:
			lon = math.atan2(pp[1], pp[0])
			lat = math.asin(pp[2])
			ny, nx = self.proj.reverse(lat, lon)
			npositions.append((ny, nx))
		return npositions

	def reverse(self, positions):
		ps = []
		for y, x in positions:
			lat, lon = self.proj.project(y, x)
			p = p2vec((lat, lon))
			ps.append(p)
		nps = self.rev.apply(ps)
		npositions = []
		for pp in nps:
			lon = math.atan2(pp[1], pp[0])
			lat = math.asin(pp[2])
			ny, nx = self.proj.reverse(lat, lon)
			npositions.append((ny, nx))
		return npositions

	# move pi1, pi2 to p1, p2
	def __init__(self, rot, H, W, F, h, w):
		self.proj = projection.Projection(W, H, F, w, h)
		self.h = h
		self.w = w
		self.H = H
		self.W = W
		self.F = F
		self.rot = rot
		self.rev = rot.inv()

	def magnitude(self):
		aa = self.rot.as_rotvec()
		angle = np.linalg.norm(aa)
		return angle**2

	def serialize(self):
		q = self.rot.as_quat()
		return json.dumps({"rot" : list(q), "h" : self.h, "w" : self.w, "H" : self.H, "W" : self.W, "F" : self.F})

	@staticmethod
	def deserialize(ser):
		s = json.loads(ser)
		q = s["rot"]
		h = s["h"]
		w = s["w"]
		H = s["H"]
		W = s["W"]
		F = s["F"]
		rot = Rotation.from_quat(np.array(q))
		return Movement(rot, H, W, F, h, w)

	@staticmethod
	def build(pi1, pi2, p1, p2, cd, debug=False):
		vi1 = p2vec(pi1)
		vi2 = p2vec(pi2)
		v1 = p2vec(p1)
		v2 = p2vec(p2)

		axis1 = vecmul(vi1, v1)
		angle1 = vecangle(vi1, v1)

		if angle1 == 0:
			rot1 = Rotation.from_rotvec([0,0,0])
		else:
			axis1 = axis1 / np.linalg.norm(axis1)
			rot1 = Rotation.from_rotvec(angle1 * axis1)

		vi1_2 = rot1.apply([vi1])[0]
		vi2_2 = rot1.apply([vi2])[0]
		if debug:
			print("angle1 = ", angle1)
		assert(vecangle(vi1_2, v1) < 1*math.pi/180)

		axis2 = v1
		axis2 = axis2 / np.linalg.norm(axis2)

		baxi = scmul(axis2, vi2_2)
		bax  = scmul(axis2, vi2)
		if debug:
			print(baxi, bax)
		assert(abs(bax - baxi) < 5e-2)
		ba = axis2 * (bax + baxi) / 2

		di = vi2_2 - ba
		d = v2 - ba
		angle2 = vecangle(di, d)
		if debug:
			print("angle2 = ", angle2)
		if scmul(v1, vecmul(di, d)) < 0:
			angle2 = -angle2

		if angle2 == 0:
			rot2 = Rotation.from_rotvec([0,0,0])
		else:
			rot2 = Rotation.from_rotvec(angle2 * axis2)

		vi1_3 = rot2.apply([vi1_2])[0]
		vi2_3 = rot2.apply([vi2_2])[0]
		
		assert(vecangle(vi1_3, v1) < 1*math.pi/180)
		assert(vecangle(vi2_3, v2) < 1*math.pi/180)

		rot = rot2 * rot1

		vi1_4 = rot.apply([vi1])[0]
		vi2_4 = rot.apply([vi2])[0]
		
		assert(vecangle(vi1_4, v1) < 0.7*math.pi/180)
		assert(vecangle(vi2_4, v2) < 0.7*math.pi/180)

		return Movement(rot, cd["H"], cd["W"], cd["F"], cd["h"], cd["w"])

	@staticmethod
	def average(ts, percent=100):
		axises = np.zeros((len(ts), 3))
		for i in range(len(ts)):
			t = ts[i]
			aa = t.rot.as_rotvec()
			axises[i,0:3] = aa
		#print("axises", axises)
		rotvec = np.average(axises, axis=0)
		dists = []
		if percent < 100:
			for i in range(len(ts)):
				daxis = axises[i] - rotvec
				dl = np.sum(daxis*daxis)**0.5
				dists.append((dl, axises[i]))
				print(dl)
			dists.sort(key=lambda item : item[0])
			dists = dists[:int(percent * len(dists) / 100)]
			rotvec = np.zeros((3,))
			for _,axis in dists:
				rotvec += axis
			rotvec /= len(dists)
		rot = Rotation.from_rotvec(rotvec)
		t = Movement(rot, ts[0].H, ts[0].W, ts[0].F, ts[0].h, ts[0].w)
		return t

