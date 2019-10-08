#!/bin/bash
#!

cd simulation2
rm *.???
sbatch --dependency=singleton --job-name=surface submit_p3.slurm
cd -

cd simulation3
rm *.???
sbatch --dependency=singleton --job-name=surface submit_p3.slurm
cd -

cd simulation4
rm *.???
sbatch --dependency=singleton --job-name=surface submit_p3.slurm
cd -

cd simulation5
rm *.???
sbatch --dependency=singleton --job-name=surface submit_p3.slurm
cd -
