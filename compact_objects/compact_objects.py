import compact_objects.detect
import compact_objects.cut

import usage

commands = {
	"detect" : (compact_objects.detect.run, "detect compact objects"),
	"cut"    : (compact_objects.cut.run, "cut compact objects"),
}

def run(argv):
	usage.run(argv, "compact_objects", commands, autohelp=True)
