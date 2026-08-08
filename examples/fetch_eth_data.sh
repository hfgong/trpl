#!/usr/bin/env bash
# Download the ETH "BIWI Walking Pedestrians" trajectories (world coords) from
# the OpenTraj mirror into data/eth/. Not committed (external dataset).
set -e
cd "$(dirname "$0")/.."
mkdir -p data/eth
BASE="https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/ETH/seq_eth"
curl -fsSL "$BASE/obsmat.txt" -o data/eth/obsmat.txt
curl -fsSL "$BASE/H.txt"      -o data/eth/H.txt
echo "Downloaded ETH data to data/eth/  ($(wc -l < data/eth/obsmat.txt) rows)"
