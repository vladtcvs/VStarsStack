import calibration.dark
import calibration.flat

import usage

commands = {
	"dark"     : (calibration.dark.run,     "dark frames handling"),
	"flat"     : (calibration.flat.run,     "flat frames handling"),
}

def run(argv):
	usage.run(argv, "calibration", commands, autohelp=True)
