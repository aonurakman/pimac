# PIMAC

Private experimentation repo for PIMAC and the compact benchmark tasks around it.

The paper-facing method is PC3D (`algorithms/pimac_v6.py`). The current core tasks are hard dynamic
Spread, LBF, and RWARE. `smacv2_task/` contains the isolated fixed-roster procedural SMACv2 runner.
IPPO, MAPPO, PIC-MAPPO, PC3D, QMIX, and MIPI form the planned SMACv2 comparison and support its
native legal-action masks; IQL, VDN, and historical PIMAC variants are excluded. Existing task
runners do not pass these optional masks, so their execution and result semantics are unchanged.

SMACv2 runs in an isolated Linux container with a repo-local, bind-mounted SC2 4.10 data directory;
see `smacv2_task/README.md`. No system-wide StarCraft installation is required.
