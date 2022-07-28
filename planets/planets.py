import planets.buildmap
import planets.configure
import usage

commands = {
	"configure" : (planets.configure.run, "configure planets in project"),
	"buildmap" : (planets.buildmap.run, "build planet surface map"),
}

def run(argv):
	usage.run(argv, "planets", commands, autohelp=True)
