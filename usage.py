#
# Copyright (c) 2022 Vladislav Tsendrovskii
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

prgname = "process.py"

def setprogname(name):
	global prgname
	prgname = name

def usage(base, commands, message):
	print("Usage: %s %s command ..." % (prgname, base))
	if message is not None:
		print("")
		print(message)
	print("")
	print("Commands:\n")
	for cmd in commands:
		if cmd != "*":
			if len(commands[cmd]) >= 3:
				print("%s - %s\n\t%s %s %s %s\n" % (cmd, commands[cmd][1], prgname, base, cmd, commands[cmd][2]))
			else:
				print("%s - %s\n\t%s %s %s ...\n" % (cmd, commands[cmd][1], prgname, base, cmd))
		else:
			if len(commands[cmd]) >= 3:
				print("(default) - %s\n\t%s %s %s\n" % (commands[cmd][1], prgname, base, commands[cmd][2]))
			else:
				print("(default) - %s\n\t%s %s ...\n" % (commands[cmd][1], prgname, base))

	print("help - print usage")
	print("\t%s %s [help]\n" % (prgname, base))

def run(argv, base, commands, message=None, autohelp=False):
	if (autohelp and len(argv) == 0) or (len(argv) > 0 and argv[0] == "help"):
		usage(base, commands, message)
		return

	if len(argv) > 0:
		cmd = argv[0]
	else:
		cmd = "*"

	if cmd not in commands:
		cmd = "*"
		commands[cmd][0](argv)
	else:
		commands[cmd][0](argv[1:])

