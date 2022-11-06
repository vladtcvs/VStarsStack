#
# Copyright (c) 2022 Vladislav Tsendrovskii
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import math
import json
import numpy as np
import math
from scipy.spatial.transform import Rotation

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

	def apply(self, positions, proj):
		ps = []
		for y, x in positions:
			lat, lon = proj.project(y, x)
			p = p2vec((lat, lon))
			ps.append(p)
		nps = self.rot.apply(ps)
		npositions = []
		for pp in nps:
			lon = math.atan2(pp[1], pp[0])
			lat = math.asin(pp[2])
			ny, nx = proj.reverse(lat, lon)
			npositions.append((ny, nx))
		return npositions

	def reverse(self, positions, proj):
		ps = []
		for y, x in positions:
			lat, lon = proj.project(y, x)
			p = p2vec((lat, lon))
			ps.append(p)
		nps = self.rev.apply(ps)
		npositions = []
		for pp in nps:
			lon = math.atan2(pp[1], pp[0])
			lat = math.asin(pp[2])
			ny, nx = proj.reverse(lat, lon)
			npositions.append((ny, nx))
		return npositions

	def __init__(self, rot):
		self.rot = rot
		self.rev = rot.inv()

	def magnitude(self):
		aa = self.rot.as_rotvec()
		angle = np.linalg.norm(aa)
		return angle**2

	def serialize(self):
		q = self.rot.as_quat()
		return json.dumps({"rot" : list(q)})

	@staticmethod
	def deserialize(ser):
		s = json.loads(ser)
		q = s["rot"]
		rot = Rotation.from_quat(np.array(q))
		return Movement(rot)


	# move pi1, pi2 to p1, p2
	@staticmethod
	def build(point1_from, point2_from, point1_to, point2_to, debug=False):
		v1_from = p2vec(point1_from)
		v2_from = p2vec(point2_from)
		v1_to = p2vec(point1_to)
		v2_to = p2vec(point2_to)

		dangle_from = vecangle(v1_from, v2_from)
		dangle_to = vecangle(v1_to, v2_to)
		assert abs(dangle_from - dangle_to) < 1*math.pi/180

		# build rotation1, which moves v1_from to v1_to
		axis1 = vecmul(v1_from, v1_to)
		angle1 = vecangle(v1_from, v1_to)

		if angle1 == 0:
			rot1 = Rotation.from_rotvec([0,0,0])
		else:
			axis1 = axis1 / np.linalg.norm(axis1)
			rot1 = Rotation.from_rotvec(angle1 * axis1)

		# intermidiate position of v1 and v2 vectors
		# v1_int must be equal to v1_to
		v1_int = rot1.apply([v1_from])[0]
		v2_int = rot1.apply([v2_from])[0]

		v1_int = v1_int / np.linalg.norm(v1_int)
		v2_int = v2_int / np.linalg.norm(v2_int)
		
		if debug:
			print("angle1 = ", angle1)
		assert(vecangle(v1_int, v1_to) < 1e-5)

		# build rotation2, which moves v2_int to v2_to
		axis2 = v1_to

		v2proj_int = scmul(axis2, v2_int)
		v2proj_to = scmul(axis2, v2_to)

		v2ort_int = v2_int - axis2 * v2proj_int
		v2ort_to = v2_to - axis2 * v2proj_to

		angle2 = vecangle(v2ort_int, v2ort_to)
		
		ort = vecmul(v2ort_int, v2ort_to)
		if scmul(ort, axis2) < 0:
			angle2 = -angle2

		if angle2 == 0:
			rot2 = Rotation.from_rotvec([0,0,0])
		else:
			rot2 = Rotation.from_rotvec(angle2 * axis2)

		v1_res = rot2.apply([v1_int])[0]
		v2_res = rot2.apply([v2_int])[0]
		
		assert(vecangle(v1_res, v1_to) < 1*math.pi/180)
		assert(vecangle(v2_res, v2_to) < 1*math.pi/180)

		rot = rot2 * rot1

		v1_res = rot.apply([v1_from])[0]
		v2_res = rot.apply([v2_from])[0]
		
		assert(vecangle(v1_res, v1_to) < 0.7*math.pi/180)
		assert(vecangle(v2_res, v2_to) < 0.7*math.pi/180)

		return Movement(rot)

	@staticmethod
	def average(ts, percent=100):
		axises = np.zeros((len(ts), 3))
		for i in range(len(ts)):
			t = ts[i]
			aa = t.rot.as_rotvec()
			axises[i,0:3] = aa
		#print("axises", axises)
		if percent == 100:
			rotvec = np.average(axises, axis=0)
		else:
			rotvec = np.average(axises, axis=0)
			dists = []
			for i in range(len(ts)):
				daxis = axises[i] - rotvec
				dl = np.sum(daxis*daxis)**0.5
				dists.append((dl, axises[i]))
#				print(dl)
			dists.sort(key=lambda item : item[0])
			num = max(1, math.ceil(percent * len(dists) / 100))
			dists = dists[:num]
			rotvec = np.zeros((3,))
			for _,axis in dists:
				rotvec += axis
			rotvec /= len(dists)
		rot = Rotation.from_rotvec(rotvec)
		t = Movement(rot)
		return t
