#!/bin/sh

dir=$1

python3 readnef.py "$dir/original" "$dir/npy"
python3 detect_stars.py "$dir/npy" "$dir/stars"
python3 describe.py "$dir/stars" "$dir/desc"
python3 match.py "$dir/desc"
python3 net.py "$dir/desc" "$dir/net.json"
python3 cluster.py "$dir/net.json" "$dir/clusters.json"
python3 calculate_shift.py "$dir/clusters.json" "$dir/shifts.json"
python3 shift.py "$dir/npy" "$dir/shifts.json" "$dir/shifted"
python3 merge.py "$dir/shifted/" "$dir/summary.npy"
python3 convert2.py "$dir/summary.npy" "$dir/summary.tiff"

