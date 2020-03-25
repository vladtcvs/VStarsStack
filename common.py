import os

def listfiles(path, ext):
	images = []
	for f in os.listdir(path):
		filename = os.path.abspath(os.path.join(path, f))
		if not os.path.isfile(filename) or f[-len(ext):].lower() != ext:
			continue
		name = os.path.splitext(f)[0]
		images.append((name, filename))
	images.sort(key=lambda item : item[0])
	return images

def length(vec):
	return (vec[0]**2+vec[1]**2)**0.5

def norm(vec):
	l = (vec[0]**2+vec[1]**2)**0.5
	return (vec[0] / l, vec[1] / l)	

