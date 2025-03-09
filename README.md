# TYK2 Active Learning

The goal is to take the data from (Optimizing active learning for free energy calculations)[https://www.sciencedirect.com/science/article/pii/S2667318522000204] and regenerate vacuum, solvent and complex RBFE calculations using (Timemachine)[https://github.com/proteneer/timemachine], the package used within the paper. The goal is to use RBFE rather than the no longer developed RABFE method to verify that the active learning is applicable to RBFE.

This is to be done on spare GPUs I have, which may be prohibitively slow to do in practice. The computations will be run on different machines, GPUs, CPUs, cuda architectures, etc, so the results may not be completely reproducible, but all of the scripts will be as similar as possible and as much information as reasonable will be tracked for each computation.

Docked poses come from https://zenodo.org/records/13759490.

## Differences from Original Predictions

- Open Forcefield 2.2.0 instead of 1.1.0
- Dynamic lambda schedule, up to 24 windows attempting to have a min BAR overlap between windows of 0.333.
- Single Topology rather than Dual Topology


## Known issues

- Re-using AM1 charges from original predictions, not as robust as AM1ELF-10 charges which is Timemachine's current default.
- Open Forcefield 2.2.0 appears to reduce the performance of some RBFE predictions, 2.0.0 is preferred by Timemachine.
- Alignment of initial ligands were performed in with the expectation of a dual topology approach, alignment may be poor in some cases.
- Region around ligands for water sampling is not adjusted for each complex leg, may be too small in some cases.

## Acknowledgments
* Jayanth Shankar - Contributed several GPUs over the course of the project, not to speak of the entertaining conversations over the year(s).
