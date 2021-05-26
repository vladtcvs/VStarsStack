import math

use_sphere = True
camerad = {
        "H" : 15.4,
        "W" : 23.1,
        "F" : 250,
        "h" : 1542,
        "w" : 2322
}

stars = {
	"match" : {
		"threshold_value" : 2,
		"threshold_num" : 0.3,
	},
}

def __vignetting(cosa):
	return (1/(9.0*math.acos(cosa)**2+1)+math.cos(math.acos(cosa)*1.8)**4)/2

vignetting = {
	"function" : __vignetting,
}

distorsion = {
	"a" : -0.228116,
	"b" : 0.034905,
	"c" : 1.012672
}

