#!/bin/bash
BASE_URL="https://scampsdataset.blob.core.windows.net/scamps/SCAMPS_"
OUT_DIR="../data/SCAMPS"

mkdir -p $OUT_DIR
cd $OUT_DIR

for i in $(seq -w 1 125); do
    FILENAME="SCAMPS_${i}.tar.gz"
    wget "${BASE_URL}${i}.tar.gz" -O $FILENAME
done