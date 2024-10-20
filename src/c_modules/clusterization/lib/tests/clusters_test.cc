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

#include <clusters.hpp>
#include <gtest/gtest.h>

TEST(clusters_test, test_1)
{
    std::vector<match_s> matches = {{{0,0}, {1,0}}};
    auto clusters = build_clusters(matches);
    ASSERT_EQ(clusters.size(), 1);
    auto cluster_1 = clusters[0];
    ASSERT_EQ(cluster_1.items.size(), 2);
    ASSERT_EQ(cluster_1.items[0].frame_id, 0);
    ASSERT_EQ(cluster_1.items[0].keypoint_id, 0);
    ASSERT_EQ(cluster_1.items[1].frame_id, 1);
    ASSERT_EQ(cluster_1.items[1].keypoint_id, 0);
}

TEST(clusters_test, test_2)
{
    std::vector<match_s> matches = {{{0,0}, {1,0}}, {{1,0}, {0,0}}};
    auto clusters = build_clusters(matches);
    ASSERT_EQ(clusters.size(), 1);
    auto cluster_1 = clusters[0];
    ASSERT_EQ(cluster_1.items.size(), 2);
    ASSERT_EQ(cluster_1.items[0].frame_id, 0);
    ASSERT_EQ(cluster_1.items[0].keypoint_id, 0);
    ASSERT_EQ(cluster_1.items[1].frame_id, 1);
    ASSERT_EQ(cluster_1.items[1].keypoint_id, 0);
}

TEST(clusters_test, test_3)
{
    std::vector<match_s> matches = {{{0,0}, {1,0}}, {{0,1}, {1,1}}, {{1,1},{2,1}}};
    auto clusters = build_clusters(matches);
    ASSERT_EQ(clusters.size(), 2);
    auto cluster_1 = clusters[0];
    ASSERT_EQ(cluster_1.items.size(), 2);
    ASSERT_EQ(cluster_1.items[0].frame_id, 0);
    ASSERT_EQ(cluster_1.items[0].keypoint_id, 0);
    ASSERT_EQ(cluster_1.items[1].frame_id, 1);
    ASSERT_EQ(cluster_1.items[1].keypoint_id, 0);
    auto cluster_2 = clusters[1];
    ASSERT_EQ(cluster_2.items.size(), 3);
    ASSERT_EQ(cluster_2.items[0].frame_id, 0);
    ASSERT_EQ(cluster_2.items[0].keypoint_id, 1);
    ASSERT_EQ(cluster_2.items[1].frame_id, 1);
    ASSERT_EQ(cluster_2.items[1].keypoint_id, 1);
    ASSERT_EQ(cluster_2.items[2].frame_id, 2);
    ASSERT_EQ(cluster_2.items[2].keypoint_id, 1);
}

TEST(clusters_test, test_4)
{
    std::vector<match_s> matches = {{{0,0}, {1,0}}, {{0,0},{0,0}}, {{0,1}, {1,1}}, {{2,1},{1,1}}};
    auto clusters = build_clusters(matches);
    ASSERT_EQ(clusters.size(), 2);
    auto cluster_1 = clusters[0];
    ASSERT_EQ(cluster_1.items.size(), 2);
    ASSERT_EQ(cluster_1.items[0].frame_id, 0);
    ASSERT_EQ(cluster_1.items[0].keypoint_id, 0);
    ASSERT_EQ(cluster_1.items[1].frame_id, 1);
    ASSERT_EQ(cluster_1.items[1].keypoint_id, 0);
    auto cluster_2 = clusters[1];
    ASSERT_EQ(cluster_2.items.size(), 3);
    ASSERT_EQ(cluster_2.items[0].frame_id, 0);
    ASSERT_EQ(cluster_2.items[0].keypoint_id, 1);
    ASSERT_EQ(cluster_2.items[1].frame_id, 1);
    ASSERT_EQ(cluster_2.items[1].keypoint_id, 1);
    ASSERT_EQ(cluster_2.items[2].frame_id, 2);
    ASSERT_EQ(cluster_2.items[2].keypoint_id, 1);
}
