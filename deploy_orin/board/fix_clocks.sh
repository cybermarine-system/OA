#!/bin/bash
set -e
sudo nvpmodel -q | grep -q "25W" || { echo "not 25W"; exit 1; }
sudo jetson_clocks && sudo jetson_clocks --fan
CPU=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
GPU=$(sudo jetson_clocks --show | grep -oP 'GPU MinFreq=\K[0-9]+')
[ "$CPU" = "1344000" ] && [ "$GPU" = "918000000" ] || { echo "clock cap mismatch: cpu=$CPU gpu=$GPU"; exit 1; }
echo "clocks fixed: cpu=$CPU gpu=$GPU"
