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

import math
from typing import Generator
import vstarstack.library.common
import vstarstack.library.data
import vstarstack.library.merge.simple_mean

def remove_dark(dataframe : vstarstack.library.data.DataFrame,
                dark : vstarstack.library.data.DataFrame):
    """Remove dark from image"""
    dark_channel_name = None
    if "L" in dark.get_channels():
        dark_channel_name = "L"
    else:
        for channel in dark.get_channels():
            if dark.get_channel_option(channel, "brightness"):
                dark_channel_name = channel
                break

    if dark_channel_name is None:
        print("Can not find brightness channel, skip")
        return None

    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if dataframe.get_channel_option(channel, "weight"):
            continue
        if not dataframe.get_channel_option(channel, "brightness"):
            print(f"Skipping {channel}, not brightness")
            continue
        if dataframe.get_channel_option(channel, "dark-removed"):
            print(f"Skipping {channel}, dark already removed")
            continue

        if channel in dark.get_channels():
            dark_layer, _ = dark.get_channel(channel)
        else:
            dark_layer, _ = dark.get_channel(dark_channel_name)

        image = image - dark_layer
        opts["dark-removed"] = True
        dataframe.replace_channel(image, channel, **opts)
    return dataframe

class TemperatureIndex:
    def __init__(self, delta_temperature : float, basic_temperature : float):
        self.dt = delta_temperature
        self.bt = basic_temperature

    def temperature_to_index(self, temperature : float) -> int | None:
        if temperature is None or self.bt is None or self.dt is None:
            return None
        return math.floor((temperature - self.bt) / self.dt + 0.5)

    def index_to_temperature(self, index : int) -> float:
        if index is None or self.bt is None or self.dt is None:
            return None
        return index * self.dt + self.bt

class FilterSource(vstarstack.library.common.IImageSource):
    """Filter sources with exposure/gain/temperature"""
    def __init__(self, source : vstarstack.library.common.IImageSource,
                       exposure : float,
                       gain : float,
                       temperature : float | None,
                       temperature_indexer : TemperatureIndex):
        self.temperature_indexer = temperature_indexer
        self.source = source
        self.exposure = exposure
        self.gain = gain
        self.temperature = temperature
        self.temperature_idx = self.temperature_indexer.temperature_to_index(self.temperature)

    def items(self) -> Generator[vstarstack.library.data.DataFrame, None, None]:
        for df in self.source.items():
            exposure = df.get_parameter("exposure")
            gain = df.get_parameter("gain")
            temperature = df.get_parameter("temperature")
            temperature_idx = self.temperature_indexer.temperature_to_index(temperature)
            if exposure == self.exposure and gain == self.gain and temperature_idx == self.temperature_idx:
                yield df

    def empty(self) -> bool:
        # TODO: better detect if there are matched df in source list
        return self.source.empty()

def prepare_darks(images : vstarstack.library.common.IImageSource,
                  basic_temperature : float | None,
                  delta_temperature : float | None) -> list:
    """Build dark frame"""
    parameters = set()
    indexer = TemperatureIndex(delta_temperature, basic_temperature)
    for df in images.items():
        exposure = df.get_parameter("exposure")
        gain = df.get_parameter("gain")
        temperature = df.get_parameter("temperature")
        index = indexer.temperature_to_index(temperature)
        parameters.add((exposure, gain, index))

    darks = []
    for exposure, gain, index in parameters:
        image_source  = FilterSource(images, exposure, gain, temperature, indexer)
        dark = vstarstack.library.merge.simple_mean.mean(image_source)
        dark.add_parameter(exposure, "exposure")
        dark.add_parameter(gain, "gain")
        temperature = indexer.index_to_temperature(index)
        if temperature is not None:
            dark.add_parameter(temperature, "temperature")
        darks.append((exposure, gain, temperature, dark))
    return darks
