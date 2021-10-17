import stars.detect
import stars.describe
import stars.match
import stars.net
import stars.cluster
import stars.lonlat
import stars.process
import stars.configure
import sys
import usage

commands = {
	"config"   : (stars.configure.run, "configure stars pipeline"),
	"detect"   : (stars.detect.run, "detect stars"),
	"lonlat"   : (stars.lonlat.run, "fill longitude and latitude"),
	"describe" : (stars.describe.run, "find descriptions for each image"),
	"match"    : (stars.match.run, "match stars between images"),
	"net"      : (stars.net.run, "build net of matching"),
	"cluster"  : (stars.cluster.run, "find matching stars clusters between images"),
	"process"  : (stars.process.run, "run process - all commands above"),
}

def run(argv):
	usage.run(argv, "stars", commands, autohelp=True)

