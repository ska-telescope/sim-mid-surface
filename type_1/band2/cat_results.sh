#!/bin/bash
#!

cat */*_?.csv | head -1 > surface_simulation_results.csv
cat */*_?.csv | grep -v context >> surface_simulation_results.csv
cat */*/*_?.csv | grep -v context >> surface_simulation_results.csv
