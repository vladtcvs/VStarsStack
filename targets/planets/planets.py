import targets.planets.buildmap
import targets.planets.configure
import usage

commands = {
	"configure" : (targets.planets.configure.run, "configure planets in project"),
	"buildmap" : (targets.planets.buildmap.run, "build planet surface map"),
}

def run(argv):
	usage.run(argv, "planets", commands, autohelp=True)
