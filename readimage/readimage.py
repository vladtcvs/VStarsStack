import usage
import readimage.nef
import readimage.jpeg

commands = {
	"nef"   : (readimage.nef.run, "read Nikon NEF"),
        "jpeg"  : (readimage.jpeg.run, "read JPEG images"),
}

def run(argv):
	usage.run(argv, "readimage", commands)

