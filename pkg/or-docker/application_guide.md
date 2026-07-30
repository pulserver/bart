# BART on OpenRecon Handbook

The Berkeley Advanced Reconstruction Toolbox (BART) toolbox
is a free and open-source image-reconstruction framework
for Computational Magnetic Resonance Imaging.

BART is packaged here as an OpenRecon application.
It can be used with standard Cartesian FLASH sequences.

To customize the reconstruction, a script can be placed at
`CustomerSeq/bart/eisbaer/reco.sh`.
This replaces the included reconstruction script,
which, for reference, can be found in the BART folder
from which this package was built, at scripts/or-reco.sh.
