import os.path
import exifread

tags_names = {
	"shutter" : [("EXIF ExposureTime", 0)],
	"iso" : [("EXIF ISOSpeedRatings", 0), ("MakerNote ISOSetting", 1)],
}

def read_tags(filename):
	f = open(filename, 'rb')
	tags = exifread.process_file(f)
	f.close()

#	for key in tags:
#		print(key, tags[key])

	res = {}
	for tn in tags_names:
		for name, id in tags_names[tn]:
			if name in tags:
				res[tn] = float(tags[name].values[id])
				break

	print(res)
	return res

