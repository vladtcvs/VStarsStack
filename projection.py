import math

class Projection(object):

	def __init__(self, W, H, F, w, h):
		self.H = H
		self.W = W
		self.h = h
		self.w = w
		self.F = F
		self.k = (self.H/self.h + self.W/self.w) / 2

	def project(self, y, x):
		X = (x - self.w / 2) * self.k
		Y = (y - self.h / 2) * self.k
		lon = math.atan(X / self.F)
		lat = math.atan(Y * math.cos(lon) / self.F)
		return lat, lon

	def reverse(self, lat, lon):
		X = self.F * math.tan(lon)
		Y = self.F * math.tan(lat) / math.cos(lon)
		x = X / self.k + self.w / 2
		y = Y / self.k + self.h / 2
		return y, x

