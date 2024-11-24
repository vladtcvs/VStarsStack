#!/usr/bin/env python3

#
# Copyright (c) 2024 Vladislav Tsendrovskii
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


import sys
import logging
import vstarstack.tool.common
import vstarstack.tool.process
import vstarstack.tool.cfg
import vstarstack.tool.usage

def main():
    program_project = vstarstack.tool.cfg.get_project()

    if vstarstack.tool.cfg.get_param("autocomplete", bool, False):
        program_argv = [item for item in sys.argv[2:] if item[:2] != "--"]
        commands = vstarstack.tool.process.commands
        variants = vstarstack.tool.usage.autocompletion(commands, program_argv)
        for variant in variants:
            print(variant)
    else:
        loglevel = vstarstack.tool.cfg.get_param("log", str, "INFO")
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        logging.basicConfig(level=numeric_level)

        program_argv = [item for item in sys.argv[1:] if item[:2] != "--"]
        vstarstack.tool.usage.run(program_project,
                                  program_argv,
                                  "",
                                  vstarstack.tool.process.commands)
