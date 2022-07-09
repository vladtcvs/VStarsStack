import usage
import readimage.nef
import readimage.jpeg
import readimage.ser

commands = {
	"nef"   : (readimage.nef.run, "read Nikon NEF"),
    "jpeg"  : (readimage.jpeg.run, "read JPEG images"),
	"ser"	: (readimage.ser.run, "read SER images"),
}

def run(argv):
	usage.run(argv, "readimage", commands, autohelp=True)
