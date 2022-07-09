import usage
import debayer.yuv422
import debayer.bayer

commands = {
	"yuv422" : (debayer.yuv422.run, "Consider RAW as YUV with 422 subsampling"),
    "bayer"  : (debayer.bayer.run, "Consider RAW as Bayer masked image")
}

def run(argv):
	usage.run(argv, "debayer", commands, autohelp=True)
