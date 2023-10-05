#!/bin/bash

for i in `seq 0 0.5 4`; do 
  echo "Run $i -th iteration"
  python3 reproduce.py "$i"

done