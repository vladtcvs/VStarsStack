import usage
import shift.select_shift
import shift.apply_shift

commands = {
	"select-shift" : (shift.select_shift.run, "Select base image and shift", "shifts.json shift.json"),
	"apply-shift"  : (shift.apply_shift.run, "Apply selected shifts", "shift.json npy/ shifted/"),
}

def run(argv):
	usage.run(argv, "shift", commands)
