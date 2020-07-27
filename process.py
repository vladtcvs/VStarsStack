import planet.planet
import stars.stars

import readnef
import merge
import shift

import sys


def usage():
	print("process command ...")
	print("Commands: ")
	print("\treadnef - read nikon NEF to npy")
	print("\tplanet - commands for processing planet images")
	print("\tstars - commands for processing stars images")
	print("\tshift - move and rotate images to match them")
	print("\tmerge - merge images")
	print("\thelp - pring help")

def run(argv):
	cmd = argv[0]
	if cmd == "readnef":
		readnef.run(argv[1:])
	elif cmd == "planet":
		planet.planet.run(argv[1:])
	elif cmd == "stars":
		stars.stars.run(argv[1:])
	elif cmd == "shift":
		shift.run(argv[1:])
	elif cmd == "merge":
		merge.run(argv[1:])
	elif cmd == "help":
		usage()
	else:
		print("Unknown command", cmd)

if __name__ == "__main__":
	run(sys.argv[1:])

