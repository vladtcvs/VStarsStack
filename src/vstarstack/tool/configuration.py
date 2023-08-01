#
# Copyright (c) 2023 Vladislav Tsendrovskii
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

class Configuration:
    """Build config structure from declaration"""

    def __init__(self, decl : dict):
        self.fields = {}
        self.used_modules = None
        self.builtin_modules = []
        self.listed_modules = []
        if "use_modules" in decl:
            self.used_modules = decl["use_modules"][1]
        for key, value in decl.items():
            if key == "use_modules":
                continue
            if isinstance(value, dict):
                self.listed_modules.append(key)
                self.builtin_modules.append(key)
                subconfig = Configuration(value)
                setattr(self, key, subconfig)
            elif isinstance(value, tuple):
                item_type = value[0]
                if item_type == "module":
                    self.listed_modules.append(key)
                    module_decl = value[1]
                    if isinstance(module_decl, dict):
                        subconfig = Configuration(module_decl)
                        setattr(self, key, subconfig)
                    else:
                        setattr(self, key, module_decl)
                else:
                    self.fields[key] = {"type":item_type, "options" : {}}
                    if len(value) > 2:
                        self.fields[key]["options"] = value[2]

                    default_value = value[1]
                    setattr(self, key, item_type(default_value))
        if self.used_modules is None:
            self.used_modules = list(self.listed_modules)

    def write_configuration(self):
        """Generate configuration file"""
        data = {}
        for key, _ in self.fields.items():
            data[key] = getattr(self, key)
        for key in self.used_modules + self.builtin_modules:
            data[key] = getattr(self, key).write_configuration()
        return data

    def load_configuration(self, saved) -> bool:
        """Load configuration from file"""
        updated = False
        self.used_modules = []
        for key, _ in self.fields.items():
            if key in saved:
                setattr(self, key, saved[key])
            else:
                updated = True
        for key in self.listed_modules:
            if key not in saved:
                continue
            updated_sub = getattr(self, key).load_configuration(saved[key])
            updated = updated or updated_sub
            if key not in self.builtin_modules:
                self.used_modules.append(key)
        return updated

    def enable_module(self, module):
        """Enable module"""
        if module not in self.used_modules:
            self.used_modules.append(module)
