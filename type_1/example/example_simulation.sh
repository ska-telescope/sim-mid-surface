#!/bin/bash
#!
python ../surface_simulation_elevation.py --context s3sky --rmax 1e5 --flux_limit 0.003 \
 --show True --elevation_sampling 1.0 --declination -45 \
--vp_directory /mnt/storage-ssd/tim/Code/sim-mid-surface/beams/interpolated/ \
--seed 18051955  --band B2 --pbtype MID_FEKO_B2 --memory 32  --integration_time 30 --use_agg True \
--time_chunk 30 --time_range -6 6  | tee surface_simulation.log
