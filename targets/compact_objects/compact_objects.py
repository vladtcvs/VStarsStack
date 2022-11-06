import targets.compact_objects.detect
import targets.compact_objects.cut
import targets.compact_objects.configure

import usage

commands = {
	"config"   : (targets.compact_objects.configure.run, "configure compact_objects pipeline"),
	"detect" : (targets.compact_objects.detect.run, "detect compact objects"),
	"cut"    : (targets.compact_objects.cut.run, "cut compact objects"),
}

def run(argv):
	usage.run(argv, "compact_objects", commands, autohelp=True)
