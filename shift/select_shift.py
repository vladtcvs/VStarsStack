import cfg
import json

from movement.sphere import Movement as ms
from movement.flat import Movement as mf

def select_base_image(infname, outfname):
	with open(infname) as f:
		data = json.load(f)

	shiftsf = data["movements"]
	if data["shift_type"] == "flat":
		Movement = mf
	elif data["shift_type"] == "sphere":
		Movement = ms
	else:
		raise Exception("Unknown shift type %s!" % data["shift_type"])

	shifts = {}
	names = []

	for name1 in shiftsf:
		shifts[name1] = {}
		names.append(name1)
		for name2 in shiftsf:
			if shiftsf[name1][name2] is None:
				continue
			shifts[name1][name2] = Movement.deserialize(shiftsf[name1][name2])

	# find image with minimal shift to other
	name0 = None
	mind2 = None
	maxc = None
	for name in names:
		c = len(shifts[name])
		print(name, c)
		if maxc is not None and c < maxc:
			continue
		if maxc is None or c > maxc:
			maxc = c
			mind2 = None
		d2 = 0
		for name2 in shifts[name]:
			shift = shifts[name][name2]
			d2 += shift.magnitude()

		if mind2 is None or d2 < mind2:
			mind2 = d2
			name0 = name

	print("Select:", name0, maxc)

	result = {}
	result["format"] = "absolute"
	result["shift_type"] = data["shift_type"]
	result["movements"] = data["movements"][name0]

	with open(outfname, "w") as f:
		json.dump(result, f, indent=4, ensure_ascii=False)

def run(argv):
	if len(argv) < 2:
		relative_name = cfg.config["paths"]["relative-shifts"]
		absolute_name = cfg.config["paths"]["absolute-shifts"]
	else:
		relative_name = argv[0]
		absolute_name = argv[1]

	select_base_image(relative_name, absolute_name)