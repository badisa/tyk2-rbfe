# TYK2 Active Learning

The goal is to take the data from (Optimizing active learning for free energy calculations)[https://www.sciencedirect.com/science/article/pii/S2667318522000204] and regenerate vacuum, solvent and complex RBFE calculations using (Timemachine)[https://github.com/proteneer/timemachine], the package used within the paper. The hope is to use RBFE rather than the no longer developed RABFE method to verify that the active learning is applicable to RBFE.

This is to be done on spare GPUs I have, which may be prohibitively slow to do in practice. The computations will be run on different machines, GPUs, CPUs, cuda architectures, etc, so the results may not be completely reproducible, but all of the scripts will be as similar as possible and as much information as reasonable will be tracked for each computation.
