"""Common movement definitions"""
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

from abc import ABC, abstractmethod

class MovementException(Exception):
    """Movement exception"""

    def __init__(self, movement_type, reason):
        Exception.__init__(
            self, f"Movement (type={movement_type}) exception: {reason}")

class Movement(ABC):
    """Interface of movements"""

    @abstractmethod
    def apply(self, positions : list, proj) -> list:
        """Apply movement to positions"""
        return []

    @abstractmethod
    def reverse(self, positions : list, proj) -> list:
        """Apply reverse movement to positions"""

    @abstractmethod
    def magnitude(self) -> float:
        """Calculate magnitude of movement"""

    @abstractmethod
    def serialize(self) -> str:
        """Serialize movement"""

    @staticmethod
    @abstractmethod
    def deserialize(ser):
        """Build movement from serialized movement description"""
