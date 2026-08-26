#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/coco}"
mkdir -p "$destination"
curl -L -o "$destination/train2017.zip" http://images.cocodataset.org/zips/train2017.zip
curl -L -o "$destination/val2017.zip" http://images.cocodataset.org/zips/val2017.zip
unzip -q "$destination/train2017.zip" -d "$destination"
unzip -q "$destination/val2017.zip" -d "$destination"

