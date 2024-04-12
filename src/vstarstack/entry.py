#!/usr/bin/env python3

import sys

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
        program_argv = [item for item in sys.argv[1:] if item[:2] != "--"]
        vstarstack.tool.usage.run(program_project,
                                  program_argv,
                                  "",
                                  vstarstack.tool.process.commands)
