#include <stdlib.h>

#include <optional>
#include <vector>
#include <map>
#include <set>

#include "clusters.hpp"

static bool operator < (const item_s &item1, const item_s& item2)
{
    if (item1.frame_id < item2.frame_id)
        return true;
    if (item1.frame_id > item2.frame_id)
        return false;
    return item1.keypoint_id < item2.keypoint_id;
}

class MatchTable {
public:
    std::map<item_s, std::set<item_s>> matches;
    std::set<item_s> items;

    MatchTable(const std::vector<match_s> &matches)
    {
        for (auto match : matches) {
            if (this->matches.find(match.item1) == this->matches.end())
                this->matches[match.item1] = std::set<item_s>();
            
            if (this->matches.find(match.item2) == this->matches.end())
                this->matches[match.item2] = std::set<item_s>();

            this->matches[match.item1].insert(match.item2);
            this->matches[match.item2].insert(match.item1);
            this->items.insert(match.item1);
            this->items.insert(match.item2);
        }
    }

    std::set<item_s> get_matches(const item_s& item) const
    {
        if (matches.find(item) == matches.end())
            return std::set<item_s>();
        return matches.find(item)->second;
    }

    std::set<item_s> get_items() const
    {
        return items;
    }
};

static std::optional<item_s> get_fresh_item(const MatchTable &matches,
                                            const std::map<item_s, int>& item_cluster_assignment)
{
    auto items = matches.get_items();
    for (auto item : items)
    {
        if (item_cluster_assignment.find(item) == item_cluster_assignment.end())
            return item;
    }
    return std::optional<item_s>();
}

struct propagate_state_s {
    item_s item;
    std::vector<item_s> neighbours;
    unsigned int index;
};

static void propagate_cluster(const item_s& item,
                              int cluster_id,
                              const MatchTable &matches,
                              std::map<item_s, int>& item_cluster_assignment)
{
    propagate_state_s initial_state;
    initial_state.item = item;
    initial_state.index = 0;
    auto nbs = matches.get_matches(item);
    std::copy(nbs.begin(), nbs.end(), std::back_inserter(initial_state.neighbours));

    std::vector<propagate_state_s> stack = {initial_state};

    item_cluster_assignment[item] = cluster_id;
    do
    {
        auto& state = stack.back();
        if (state.index >= state.neighbours.size())
        {
            stack.pop_back();
            if (stack.size() > 0)
            {
                stack.back().index++;
                continue;
            }
            else
            {
                break;
            }
        }

        item_s neighbour = state.neighbours[state.index];
        if (item_cluster_assignment.find(neighbour) != item_cluster_assignment.end())
        {
            // Already have iterated this item
            state.index++;
            continue;
        }

        propagate_state_s newstate;
        newstate.item = neighbour;
        newstate.index = 0;
        auto nbs = matches.get_matches(neighbour);
        std::copy(nbs.begin(), nbs.end(), std::back_inserter(newstate.neighbours));
        
        item_cluster_assignment[neighbour] = cluster_id;
        stack.push_back(newstate);
    } while (true);
}

std::vector<cluster_s> build_clusters(const std::vector<match_s> &matches)
{
    MatchTable table(matches);
    std::map<item_s, int> item_cluster_assignment;
    int cluster_id = 0;
    while (1) {
        std::optional<item_s> item = get_fresh_item(table, item_cluster_assignment);
        if (!item.has_value())
            break;
        propagate_cluster(item.value(), cluster_id, table, item_cluster_assignment);
        cluster_id++;
    }
    std::vector<cluster_s> clusters;
    clusters.resize(cluster_id);
    for (auto pair : item_cluster_assignment)
    {
        item_s item = pair.first;
        int id = pair.second;
        clusters[id].items.push_back(item);
    }

    std::vector<cluster_s> clusters_big;
    for (auto cluster : clusters)
        if (cluster.items.size() > 1)
            clusters_big.push_back(cluster);
    return clusters_big;
}
