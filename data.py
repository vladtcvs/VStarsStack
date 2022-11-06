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

import cfg
import zipfile
import json
import numpy as np

class DataFrame(object):
	def __init__(self, params=None, tags=None):
		if tags is None:
			self.tags = {}
		else:
			self.tags = tags
		
		if params is None:
			self.params = {}
		else:
			self.params = params

		self.links = {"weight" : {}}

		self.channels = {}
	
	def add_channel(self, data, name, **options):
		# some required options
		if "weight" not in options:
			options["weight"] = False
		if "encoded" not in options:
			options["encoded"] = False
		if "brightness" not in options:
			options["brightness"] = False

		self.channels[name] = {
			"data" : data,
			"options" : options,
		}

	def add_channel_link(self, name, linked, link_type):
		if name not in self.channels or linked not in self.channels:
			return

		if link_type not in self.links:
			self.links[link_type] = {}
		self.links[link_type][name] = linked

	def remove_channel(self, name):
		if name in self.channels:
			self.channels.pop(name)
		for linktype in self.links:
			if name in self.links[linktype]:
				self.links[linktype].pop(name)

			remove = []
			for dname in self.links[linktype]:
				if self.links[linktype][dname] == name:
					remove.append(dname)
			for dname in remove:
				self.links[linktype].pop(dname)

	def rename_channel(self, name, target):
		if name not in self.channels:
			return
		if target in self.channels:
			return
		if name == target:
			return
		self.channels[target] = self.channels[name]
		self.channels.pop(name)
		for linktype in self.links:
			if name in self.links[linktype]:
				self.links[linktype][target] = self.links[linktype][name]
				self.links[linktype].pop(name)
			for dname in self.links[linktype]:
				if self.links[linktype][dname] == name:
					self.links[linktype][dname] = target

	def add_parameter(self, value, name):
		self.params[name] = value

	def get_channel(self, channel):
		return self.channels[channel]["data"], self.channels[channel]["options"]

	def get_channels(self):
		return list(self.channels.keys())

	@staticmethod
	def _store_json(value, file):
		file.write(bytes(json.dumps(value, indent=4, ensure_ascii=False), 'utf8'))

	def store(self, fname, compress=None):
		if compress is None:
			try:
				compress = cfg.compress
			except:
				compress = True
		if compress:
			method = zipfile.ZIP_LZMA
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
			
			for channel in self.channels:
				with zf.open(channel+".npy", "w") as f:
					np.save(f, self.channels[channel]["data"])
				with zf.open(channel+".json", "w") as f:
					self._store_json(self.channels[channel]["options"], f)

	@staticmethod
	def load(fname):
		try:
			with zipfile.ZipFile(fname, "r") as zf:
				with zf.open("channels.json", "r") as f:
					channels = json.load(f)
				with zf.open("params.json", "r") as f:
					params = json.load(f)
				with zf.open("tags.json", "r") as f:
					tags = json.load(f)
				with zf.open("links.json", "r") as f:
					links = json.load(f)

				data = DataFrame(params, tags)

				for channel in channels:
					with zf.open(channel+".json", "r") as f:
						options = json.load(f)
					with zf.open(channel+".npy", "r") as f:
						data.add_channel(np.load(f), channel, **options)

				for link_type in links:
					for name in links[link_type]:
						data.add_channel_link(name, links[link_type][name], link_type)

		except Exception as e:
			print("Error reading %s" % input)
			raise e
		return data
