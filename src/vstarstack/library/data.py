"""Dataframe structure for images"""
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

import zipfile
import json
import numpy as np

class InvalidParameterException(Exception):
    """Invalid parameter in DataFrame"""

    def __init__(self, parameter, desc=""):
        Exception.__init__(self, f"Invalid parameter {parameter}: {desc}")


class DataFrame:
    """Frame with single image"""

    def __init__(self, params=None, tags=None):
        if tags is None:
            self.tags = {}
        else:
            self.tags = tags

        if params is None:
            self.params = {}
        else:
            self.params = params

        self.links = {"weight": {}}

        self.channels = {}

    def add_channel(self, data, name, **options):
        """Add channel image to dataframe"""
        # some required options
        if "weight" not in options:
            options["weight"] = False
        if "encoded" not in options:
            options["encoded"] = False
        if "brightness" not in options:
            options["brightness"] = False

        self.channels[name] = {
            "data": data,
            "options": options,
        }

    def replace_channel(self, data, name):
        """Replace channel image"""
        if name not in self.channels:
            return False
        self.channels[name]["data"] = data
        return True

    def add_channel_link(self, name, linked, link_type):
        """Create link between 2 channels"""
        if name not in self.channels or linked not in self.channels:
            return

        if link_type not in self.links:
            self.links[link_type] = {}
        self.links[link_type][name] = linked

    def remove_channel(self, name):
        """Remove channel from dataframe"""
        if name in self.channels:
            self.channels.pop(name)
        for linktype, links in self.links.items():
            if name in links:
                self.links[linktype].pop(name)

            remove = []
            for dname in links:
                if self.links[linktype][dname] == name:
                    remove.append(dname)
            for dname in remove:
                self.links[linktype].pop(dname)

    def rename_channel(self, name, target):
        """Rename channel"""
        if name not in self.channels:
            return
        if target in self.channels:
            return
        if name == target:
            return
        self.channels[target] = self.channels[name]
        self.channels.pop(name)
        for linktype, links in self.links.items():
            if name in links:
                self.links[linktype][target] = self.links[linktype][name]
                self.links[linktype].pop(name)
            for dname in links:
                if self.links[linktype][dname] == name:
                    self.links[linktype][dname] = target

    def add_parameter(self, value, name):
        """Add parameter"""
        self.params[name] = value

    def get_channel(self, channel):
        """Get channel image"""
        return self.channels[channel]["data"], self.channels[channel]["options"]

    def get_channels(self):
        """Get list of channels"""
        return list(self.channels.keys())

    @staticmethod
    def _store_json(value, file):
        file.write(bytes(json.dumps(value, indent=4, ensure_ascii=False), 'utf8'))

    def store(self, fname, compress=None):
        """Save dataframe to file"""
        if compress is None:
            compress = True

        if compress:
            method = zipfile.ZIP_BZIP2
        else:
            method = zipfile.ZIP_STORED

        with zipfile.ZipFile(fname, mode="w", compression=method) as zf:
            with zf.open("tags.json", "w") as f:
                self._store_json(self.tags, f)
            with zf.open("params.json", "w") as f:
                self._store_json(self.params, f)
            with zf.open("channels.json", "w") as f:
                self._store_json(self.get_channels(), f)
            with zf.open("links.json", "w") as f:
                self._store_json(self.links, f)

            for channel_name, channel in self.channels.items():
                with zf.open(channel_name+".npy", "w") as f:
                    np.save(f, channel["data"])
                with zf.open(channel_name+".json", "w") as f:
                    self._store_json(channel["options"], f)

    @staticmethod
    def load(fname):
        """Load dataframe from file"""
        try:
            with zipfile.ZipFile(fname, "r") as zip_file:
                with zip_file.open("channels.json", "r") as file:
                    channels = json.load(file)
                with zip_file.open("params.json", "r") as file:
                    params = json.load(file)
                with zip_file.open("tags.json", "r") as file:
                    tags = json.load(file)
                with zip_file.open("links.json", "r") as file:
                    links = json.load(file)

                data = DataFrame(params, tags)

                for channel in channels:
                    with zip_file.open(channel+".json", "r") as file:
                        options = json.load(file)
                    with zip_file.open(channel+".npy", "r") as file:
                        data.add_channel(np.load(file), channel, **options)

                for link_type in links:
                    for name in links[link_type]:
                        data.add_channel_link(
                            name, links[link_type][name], link_type)

        except Exception as excp:
            print(f"Error reading {input} : {excp}")
            raise excp
        return data
