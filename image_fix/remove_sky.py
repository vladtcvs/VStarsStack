import image_fix.remove_sky_methods.isoline
import image_fix.remove_sky_methods.gauss

import usage

commands = {
	"isoline" : (image_fix.remove_sky_methods.isoline.run, "use isoline model"),
	"gauss"   : (image_fix.remove_sky_methods.gauss.run,   "use gauss blur model"),
}

def run(argv):
	usage.run(argv, "image-fix remove-sky", commands, autohelp=True)
