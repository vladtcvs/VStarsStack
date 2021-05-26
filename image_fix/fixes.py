import image_fix.vignetting
import image_fix.distorsion
import image_fix.remove_sky

import usage

commands = {
	"distorsion" : (image_fix.distorsion.run, "fix distorsion"),
	"vignetting" : (image_fix.vignetting.run, "fix vignetting"),
	"remove-sky" : (image_fix.remove_sky.run, "remove sky"),
}

def run(argv):
	usage.run(argv, "image-fix", commands)

