#!/usr/bin/env bash

mkdir -p test_temporary_data

w=-15.456047
s=15.665024
e=-15.325491
n=15.787501

wktpoly="POLYGON(($w $s,$e $s,$e $n,$w $n,$w $s))"

python run_efast.py \
  --start-date 2022-09-07 \
  --end-date 2022-09-17 \
  --aoi-geometry "$wktpoly" \
  --s3-sensor SYN \
  --mosaic-days 100 \
  --step 2 \
  --snap-gpt-path $(which gpt)
