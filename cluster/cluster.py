import cluster.restore
import cluster.find_shift
import usage

def run(argv):
	cmd = argv[0]
	if cmd == "help":
		print("restore - restore lon & lat in cluster description")
		print("find-shift - build shifts between images")
	elif cmd == "find-shift":
		cluster.find_shift.run(argv[1:])
	elif cmd == "restore":
		cluster.restore.run(argv[1:])

