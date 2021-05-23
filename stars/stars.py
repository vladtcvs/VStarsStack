import stars.detect
import stars.describe
import stars.match
import stars.net
import stars.cluster
import stars.lonlat
import stars.process
import sys

def run(argv):
	cmd = argv[0]
	if cmd == "detect":
		stars.detect.run(argv[1:])
	elif cmd == "lonlat":
		stars.lonlat.run(argv[1:])
	elif cmd == "describe":
		stars.describe.run(argv[1:])
	elif cmd == "match":
		stars.match.run(argv[1:])
	elif cmd == "net":
		stars.net.run(argv[1:])
	elif cmd == "cluster":
		stars.cluster.run(argv[1:])
	elif cmd == "process":
		stars.process.run(argv[1:])
	else:
		print("Unknown command", cmd)

if __name__ == "__main__":
	run(sys.argv[1:])

