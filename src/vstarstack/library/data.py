"""Dataframe structure for images"""
#
# Copyright (c) 2022-2024 Vladislav Tsendrovskii
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

import logging
import zipfile
import json
from typing import Tuple, List
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)

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

    def copy(self):
        """Copy dataframe"""
        new = DataFrame(self.params, self.tags)
        for name in self.get_channels():
            channel, opts = self.get_channel(name)
            opts = dict(opts)
            
            new.add_channel(deepcopy(channel), name, **opts)
        for link_type in self.links:
            for name in self.links[link_type]:
                new.add_channel_link(name, self.links[link_type][name], link_type)
        return new

    def add_channel(self, data : np.ndarray, name : str, **options):
        """Add channel image to dataframe"""
        # some required options
        if "weight" not in options:
            options["weight"] = False
        if "encoded" not in options:
            options["encoded"] = False
        if "brightness" not in options:
            options["brightness"] = False
        if "signal" not in options:
            options["signal"] = False
        if "normed" not in options:
            options["normed"] = False

        self.channels[name] = {
            "data": data,
            "options": options,
        }

    def replace_channel(self, data : np.ndarray, name : str, **options):
        """Replace channel image"""
        if name not in self.channels:
            return False
        self.channels[name]["data"] = data
        for key in options:
            self.channels[name]["options"][key] = options[key]
        return True

    def add_channel_link(self, name : str, linked : str, link_type : str):
        """Create link between 2 channels"""
        if name not in self.channels or linked not in self.channels:
            return

        if link_type not in self.links:
            self.links[link_type] = {}
        self.links[link_type][name] = linked

    def remove_channel(self, name : str):
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

    def rename_channel(self, name : str, target : str):
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

    def add_parameter(self, value, name : str):
        """Add parameter"""
        self.params[name] = value

    def get_parameter(self, name : str):
        """
        Get parameter

        Parameters:
            name (str) - parameter name
        
        Returns:
            None - if parameter doesn't exists
            parameter value - if parameter exists
        """
        if name not in self.params:
            return None
        return self.params[name]

    def get_channel(self, channel : str) -> Tuple[np.ndarray, dict]:
        """
        Get channel image.

        Parameters:
            channel (str) - channel name

        Returns:
            tuple(image, options) - ndarray with pixel data and channel options
        """
        if channel not in self.channels:
            return None, None
        return self.channels[channel]["data"], self.channels[channel]["options"]

    def get_channels(self) -> List[str]:
        """Get list of channels"""
        return list(self.channels.keys())

    def get_channel_option(self, channel : str, option : str) -> bool | None:
        """
        Get option of channel.

        Parameters:
            channel (str) - channel name
            option (str) - option name

        Returns:
            None if channel doesn't exist
            False if option is not exist
            option value if option is exist
        """
        if channel not in self.channels:
            return None
        if option not in self.channels[channel]["options"]:
            return False
        return self.channels[channel]["options"][option]

    def get_linked_channel(self, channel : str, link_type : str):
        """
        Get linked channel

        Parameters:
            channel (str) - channel name
            link_type (str) - type of link
        
        Returns:
            None, None, None if no such link or no linked channel
            layer, opts, name if such linked channel exists
        """
        if link_type not in self.links:
            return None, None, None
        if channel not in self.links[link_type]:
            return None, None, None
        name = str(self.links[link_type][channel])
        layer, opts = self.get_channel(name)
        return layer, opts, name

    @staticmethod
    def _store_json(value, file):
        file.write(bytes(json.dumps(value, indent=4, ensure_ascii=False), 'utf8'))

    def _normalize_weight(self, data : np.ndarray, weight_max : float) -> Tuple[np.ndarray, float]:
        """Convert weight channel to (ndarray(uint16), float)"""
        if data.dtype == np.uint16:
            return data, np.amax(data)*weight_max
        data = np.clip(data, a_min=0, a_max=None)
        maxv = np.amax(data)
        data = (data*65535/maxv).astype(np.uint16)
        return data, float(maxv*weight_max)

    def store(self, fname : str, compress : bool|None = None):
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
                data = channel["data"]
                opts = channel["options"]
                if opts["weight"] == True:
                    weight_max = 1
                    if "weight_max" in opts:
                        weight_max = opts["weight_max"]
                    data, weight_max = self._normalize_weight(data, weight_max)
                    opts["weight_max"] = weight_max
                with zf.open(channel_name+".npy", "w") as f:
                    np.save(f, data)
                with zf.open(channel_name+".json", "w") as f:
                    self._store_json(opts, f)

    @staticmethod
    def load(fname : str):
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
                        content = np.load(file)
                    if "weight" in options and "weight_max" in options:
                        content = content.astype(np.float32)
                        content = content * options.pop("weight_max") / 65535
    
                    data.add_channel(content, channel, **options)

                for link_type in links:
                    for name in links[link_type]:
                        data.add_channel_link(
                            name, links[link_type][name], link_type)

        except Exception as excp:
            logger.error(f"Error reading {input} : {excp}")
            raise excp
        return data
