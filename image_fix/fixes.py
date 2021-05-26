import optical_fix.vignetting
import optical_fix.distorsion

import sys


def usage():
	print("Commands: ")
	print("\tdistorsion - fix distorsion")
	print("\tvignetting - fix vignetting")
	print("\thelp - pring help")

def run(argv):
	cmd = argv[0]
	if cmd == "distorsion":
		optical_fix.distorsion.run(argv[1:])
	elif cmd == "vignetting":
		optical_fix.vignetting.run(argv[1:])
	elif cmd == "help":
		usage()
	else:
		print("Unknown command", cmd)

if __name__ == "__main__":
	run(sys.argv[1:])

