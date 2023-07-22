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

import os
_PRGNAME = "vstarstack"

def complete_path_in_dir(dirname : str | None, prefix : str):
    paths = os.listdir(dirname)
    variants = []
    for path in paths:
        if not path.startswith(prefix):
            continue
        if dirname is not None:
            full_path = os.path.join(dirname, path)
        else:
            full_path = path
        if os.path.isdir(full_path):
            variants.append((path, True))
        else:
            variants.append((path, False))
    return sorted(variants, key=lambda item : item[0])

def autocomplete_files(prefix : str):
    """Autocomplete path"""
    pos = prefix.rfind(os.sep)
    if pos == -1:
        variants = complete_path_in_dir(None, prefix)
        comps = []
        for path, isdir in variants:
            if isdir:
                comps.append(path + os.sep)
            else:
                comps.append(path)
        return comps

    dirname = prefix[:pos]
    last = prefix[pos+1:]
    variants = complete_path_in_dir(dirname, last)
    comps = []
    for path, isdir in variants:
        if isdir:
            comps.append(os.path.join(dirname,  path + os.sep))
        else:
            comps.append(os.path.join(dirname, path))
    return comps


def autocompletion(commands : dict, argv : list):
    """Autocompletion"""
    if len(argv) == 0:
        current_input = ""
    else:
        current_input = argv[0]
    for cmd in commands:
        if cmd == current_input:
            submodule = commands[cmd][0]
            if isinstance(submodule, dict):
                return autocompletion(submodule, argv[1:])
            if len(argv) > 1:
                last = argv[-1]
            else:
                last = ""
            return autocomplete_files(last)

    variants = []
    for cmd in commands:
        if cmd.startswith(current_input):
            variants.append(cmd)
    return variants

def usage(base : str, commands : dict, message : str):
    """Display usage"""
    print(f"Usage: {_PRGNAME} {base} command ...")
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
                print(f"{cmd} - {desc}\n\t{_PRGNAME} {base} {cmd} {extra}\n")
            else:
                print(f"{cmd} - {desc}\n\t{_PRGNAME} {base} {cmd} ...\n")
        else:
            if len(commands[cmd]) >= 3:
                extra = commands[cmd][2]
                print(f"(default) - {desc}\n\t{_PRGNAME} {base} {extra}\n")
            else:
                print(f"(default) - {desc}\n\t{_PRGNAME} {base} ...\n")

    print("help - print usage")
    print(f"\t{_PRGNAME} {base} [help]\n")


def run(project, argv, base, commands, message=None):
    """Run usage"""
    if (len(argv) == 0) or (len(argv) > 0 and argv[0] == "help"):
        usage(base, commands, message)
        return

    cmd = argv[0]
    if cmd not in commands:
        print(f"Command {cmd} not found!")
        usage(base, commands, message)
        return

    submodule = commands[cmd][0]
    if isinstance(submodule, dict):
        if len(base) > 0:
            new_base = base + " " + cmd
        else:
            new_base = cmd
        run(project, argv[1:], new_base, submodule, message)
    else:
        submodule(project, argv[1:])
