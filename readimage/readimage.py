import usage
import readimage.nef
import readimage.classic
import readimage.ser

commands = {
	"nef"   : (readimage.nef.run, "read Nikon NEF"),
	"classic"  : (readimage.classic.run, "read usual images (JPG, PNG, TIFF)"),
	"ser" : (readimage.ser.run, "read SER"),
}

def run(argv):
	usage.run(argv, "readimage", commands, autohelp=True)
