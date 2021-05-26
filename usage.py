prgname = "process.py"

def usage(base, commands):
	print("Usage: %s %s command ..." % (prgname, base))
	print("Commands:\n")
	for cmd in commands:
		if len(commands[cmd]) >= 3:
			print("%s - %s\n\t%s %s %s %s\n" % (cmd, commands[cmd][1], prgname, base, cmd, commands[cmd][2]))
		else:
			print("%s - %s\n\t%s %s %s ...\n" % (cmd, commands[cmd][1], prgname, base, cmd))
	print("help - print usage\n")

def run(argv, base, commands):
	if len(argv) == 0 or argv[0] not in commands:
		usage(base, commands)
		return
	cmd = argv[0]
	commands[cmd][0](argv[1:])

