/*
 * Copyright (c) 2024 Vladislav Tsendrovskii
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <vector>

struct item_s {
    int frame_id;
    int keypoint_id;
};

struct match_s {
    item_s item1;
    item_s item2;
};

struct cluster_s {
    std::vector<item_s> items;
};

std::vector<cluster_s> build_clusters(const std::vector<match_s> &matches);
