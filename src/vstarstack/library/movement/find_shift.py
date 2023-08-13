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

from vstarstack.library.movement.basic_movement import Movement

def find_movement(movement, star_pairs):
    """Find shift, using set of stars"""
    movements = []
    for i, (s1_f, s1_t) in enumerate(star_pairs):
        for j in range(i):
            s2_f = star_pairs[j][0]
            s2_t = star_pairs[j][1]
            mov = movement.build(s1_f, s2_f, s1_t, s2_t)
            movements.append(mov)
    return movement.average(movements, 80)

def build_movements(movement : Movement, clusters : list):
    """Build movements between different images"""
    names = []
    for cluster in clusters:
        for name in cluster:
            if name not in names:
                names.append(name)

    movements = {}
    errors = []

    for name1 in names:
        movements[name1] = {}
        for name2 in names:
            if name1 == name2:
                movements[name1][name2] = movement.identity()
                continue

            stars = []
            for cluster in clusters:
                if name1 not in cluster:
                    continue
                if name2 not in cluster:
                    continue

                star_to = (cluster[name1]["lon"], cluster[name1]["lat"])
                star_from = (cluster[name2]["lon"], cluster[name2]["lat"])
                stars.append((star_from, star_to))

            if len(stars) >= 2:
                try:
                    movements[name1][name2] = find_movement(movement, stars)
                except:
                    errors.append((name1, name2))
    return movements, errors

def complete_movements(movement : Movement, movements : dict, compose : bool):
    names = set()
    for name1 in movements:
        names.add(name1)
        for name2 in movements[name1]:
            names.add(name2)
    changed = True

    while changed:
        changed = False
        created = []

        # create identity
        for name in names:
            if name not in movements:
                created.append((name, name, movement.identity()))
            elif name not in movements[name]:
                created.append((name, name, movement.identity()))

        # create inversed movements
        for name1 in movements:
            for name2 in movements[name1]:
                if name1 not in movements[name2]:
                    inversed = movements[name1][name2].inverse()
                    created.append((name2, name1, inversed))

        if len(created) == 0 and compose:
            # create composed methods
            for name1 in movements:
                for name2 in movements[name1]:
                    if name2 not in movements:
                        continue
                    for name3 in movements[name2]:
                        if name3 in movements[name1]:
                            continue
                        # create movement by composition
                        mov12 = movements[name1][name2]
                        mov23 = movements[name2][name3]
                        movement = mov12 * mov23
                        created.append((name1, name3, movement))

        for name1, name2, movement in created:
            if name1 not in movements:
                movements[name1] = {}
            movements[name1][name2] = movement
            changed = True

    return movements
