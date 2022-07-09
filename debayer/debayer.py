import usage
import debayer.yuv422

commands = {
	"yuv422" : (debayer.yuv422.run, "Consider RAW as YUV with 422 subsampling"),
}

def run(argv):
	usage.run(argv, "debayer", commands, autohelp=True)
