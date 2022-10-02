import cfg
import zipfile
import json
import numpy as np


def data_create(tags={}, params={}):
	data = {
		"channels": {},
		"meta": {
			"channels": [],
			"encoded_channels": [],
			"tags": tags,
			"params": params,
		},
	}
	return data


def data_add_channel(data, channel, name, encoded=False):
	data["channels"][name] = channel
	if name not in data["meta"]["channels"]:
		data["meta"]["channels"].append(name)
	if encoded and name not in data["meta"]["encoded_channels"]:
		data["meta"]["encoded_channels"].append(name)
	return data


def data_add_parameter(data, value, name):
	data["meta"]["params"][name] = value


def data_store(data, output, compress=None):
	if compress is None:
		compress = cfg.compress

	if compress:
		method = zipfile.ZIP_LZMA
	else:
		method = zipfile.ZIP_STORED

	with zipfile.ZipFile(output, mode="w", compression=method) as zf:
		with zf.open("meta.json", "w") as f:
			f.write(
				bytes(json.dumps(data["meta"], indent=4, ensure_ascii=False), 'utf8'))
			#json.dump(data["meta"], f, indent=4, ensure_ascii=False)
		for channel in data["channels"]:
			with zf.open(channel+".npy", "w") as f:
				np.save(f, data["channels"][channel])


def data_load(input):
	try:
		with zipfile.ZipFile(input, "r") as zf:
			with zf.open("meta.json", "r") as f:
				meta = json.load(f)
			data = data_create(meta["tags"], meta["params"])

			for channel in meta["channels"]:
				with zf.open(channel+".npy", "r") as f:
					encoded = channel in meta["encoded_channels"]
					data_add_channel(data, np.load(f), channel, encoded=encoded)
	except Exception as e:
		print("Error reading %s" % input)
		raise e
	return data

def data_remove_channel(data, name):
	if name in data["channels"]:
		data["channels"].pop(name)

	if name in data["meta"]["channels"]:
		data["meta"]["channels"].remove(name)
	if name in data["meta"]["encoded_channels"]:
		data["meta"]["encoded_channels"].remove(name)
