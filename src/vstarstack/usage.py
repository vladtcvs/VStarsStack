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

PRGNAME = "process.py"


def setprogname(name):
    """Setup program name"""
    global PRGNAME
    PRGNAME = name


def usage(base, commands, message):
    """Display usage"""
    print(f"Usage: {PRGNAME} {base} command ...")
    if message is not None:
        print("")
        print(message)
    print("")
    print("Commands:\n")
    for cmd in commands:
        desc = commands[cmd][1]
        if cmd != "*":
            if len(commands[cmd]) >= 3:
                extra = commands[cmd][2]
                print(f"{cmd} - {desc}\n\t{PRGNAME} {base} {cmd} {extra}\n")
            else:
                print(f"{cmd} - {desc}\n\t{PRGNAME} {base} {cmd}...\n")
        else:
            if len(commands[cmd]) >= 3:
                extra = commands[cmd][2]
                print(f"(default) - {desc}\n\t{PRGNAME} {base} {extra}\n")
            else:
                print(f"(default) - {desc}\n\t{PRGNAME} {base}...\n")

    print("help - print usage")
    print(f"\t{PRGNAME} {base} [help]\n")


def run(project, argv, base, commands, message=None, autohelp=False):
    """Run usage"""
    if (autohelp and len(argv) == 0) or (len(argv) > 0 and argv[0] == "help"):
        usage(base, commands, message)
        return

    if len(argv) > 0:
        cmd = argv[0]

        if cmd not in commands:
            print(f"Command {cmd} not found!")
            usage(base, commands, message)
            return
        commands[cmd][0](project, argv[1:])
    else:
        commands["*"][0](project, argv)
