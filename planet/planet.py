import planet.detect
import planet.cut
import sys

def run(argv):
	cmd = argv[0]
	if cmd == "detect":
		planet.detect.run(argv[1:])
	elif cmd == "cut":
		planet.cut.run(argv[1:])
	else:
		print("Unknown command", cmd)

if __name__ == "__main__":
	run(sys.argv[1:])

