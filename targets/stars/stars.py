import targets.stars.detect
import targets.stars.describe
import targets.stars.match
import targets.stars.net
import targets.stars.cluster
import targets.stars.lonlat
import targets.stars.process
import targets.stars.configure
import sys
import usage

commands = {
	"config"   : (targets.stars.configure.run, "configure stars pipeline"),
	"detect"   : (targets.stars.detect.run, "detect stars"),
	"lonlat"   : (targets.stars.lonlat.run, "fill longitude and latitude"),
	"describe" : (targets.stars.describe.run, "find descriptions for each image"),
	"match"    : (targets.stars.match.run, "match stars between images"),
	"net"      : (targets.stars.net.run, "build net of matching"),
	"cluster"  : (targets.stars.cluster.run, "find matching stars clusters between images"),
	"process"  : (targets.stars.process.run, "run process - all commands above"),
}

def run(argv):
	usage.run(argv, "stars", commands, autohelp=True)

