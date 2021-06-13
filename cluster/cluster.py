import cluster.restore
import cluster.find_shift
import usage

commands = {
	"find-shift" : (cluster.find_shift.run, "Find shifts from cluster file", "cluster.json shifts.json"),
	"restore"    : (cluster.restore.run, "Restore lon lat in cluster file", "cluster.json [cluster_out.json]"),
}

def run(argv):
	usage.run(argv, "cluster", commands)

