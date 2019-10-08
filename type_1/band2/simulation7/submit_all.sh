#!/bin/bash
#!

cd p15
rm *.???
sbatch --dependency=singleton --job-name=surface submit_p3.slurm
cd -

cd 0
rm *.???
sbatch --dependency=singleton --job-name=surface submit_p3.slurm
cd -

cd m15
rm *.???
sbatch --dependency=singleton --job-name=surface submit_p3.slurm
cd -

cd m30
rm *.???
sbatch --dependency=singleton --job-name=surface submit_p3.slurm
cd -

cd m45
rm *.???
sbatch --dependency=singleton --job-name=surface submit_p3.slurm
cd -

cd m60
rm *.???
sbatch --dependency=singleton --job-name=surface submit_p3.slurm
cd -

cd m75
rm *.???
sbatch --dependency=singleton --job-name=surface submit_p3.slurm
cd -
