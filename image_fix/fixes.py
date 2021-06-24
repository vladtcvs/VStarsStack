import image_fix.vignetting
import image_fix.distorsion
import image_fix.remove_sky
import image_fix.difference
import image_fix.border

import usage

commands = {
	"distorsion" : (image_fix.distorsion.run, "fix distorsion"),
	"vignetting" : (image_fix.vignetting.run, "fix vignetting"),
	"remove-sky" : (image_fix.remove_sky.run, "remove sky"),
	"difference" : (image_fix.difference.run, "substract dark frame"),
	"border"     : (image_fix.border.run,     "remove border"),
}

def run(argv):
	usage.run(argv, "image-fix", commands)

