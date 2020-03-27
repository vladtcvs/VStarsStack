#!/bin/sh

dir=$1

echo "Readnef"
#python3 readnef.py "$dir/original" "$dir/npy"

echo "Detect stars"
python3 detect_stars.py "$dir/npy" "$dir/stars"

echo "Build descriptors"
python3 describe.py "$dir/stars" "$dir/desc"

echo "Match stars"
python3 match.py "$dir/desc"

echo "Build net of stars matching"
python3 net.py "$dir/desc" "$dir/net.json"

echo "Find star matching clusters"
python3 cluster.py "$dir/net.json" "$dir/desc" "$dir/clusters.json"

echo "Calculate shifts between images"
python3 calculate_shift.py "$dir/clusters.json" "$dir/shifts.json"

echo "Make shifts"
python3 shift.py "$dir/npy" "$dir/shifts.json" "$dir/shifted"

echo "Merge images"
python3 merge.py "$dir/shifted/" "$dir/summary.npy"
#python3 convert2.py "$dir/summary.npy" "$dir/summary.tiff"

