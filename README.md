# 463-final-project
A dump of my code and data for this project.

# Usage
See `*_runner.py` in `src/` for usage. See `src/realsense/*_data/` for source data. 

# Project overview
This project was originally scoped as a re-implementation of Kadambi Et Al's 2017 "Depth sensing using geometrically constrained polarization normals." However, due to significant difficulties implementing the early building blocks of the project, as well as difficulties finding suitable libraries for many parts of Kadambi's procedure, the project was essentially scaled back to implementing Nehab's "normal+depth" fusion procedure with naive SfP as described by many previously (Atkinson, Miyazaki, etc.)

Additionally, getting normals by photometric stereo was also used to provide a point of comparison with SfP and additional input to the Nehab algorithm. 

# Citations
Papers that provided motivation for certain aspects of the code have been cited where relevant in the codebase at the of the file to the best of my knowledge/recollection. 

Principally, Smith provided the most detail in the following papers, which are all very well written and were significantly more useful and clear than any other papers I read on the topics. I only wish I had read them at the start of the project, and would encourage others trying to work on this material to start from here and work your way up to more advanced topics like in Kadambi. 

- William A. P. Smith, Ravi Ramamoorthi, and Silvia Tozza. Linear depth estimation from an uncalibrated, monocular polarisation image. pages 109–125, 2016.

- William A. P. Smith, Ravi Ramamoorthi, and Silvia Tozza. Height-from-polarisation with unknown lighting or albedo. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(12):2875–2888, 2019.

- Ye Yu and William Smith. Depth estimation meets inverse rendering for single image novel view synthesis. CVMP ’19: European Conference on Visual Media Production, pages 1–7, 12 2019.


# Project details
See `463 fp rbm.pdf` for more details.

# Future directions
I was planning on adding a disambiguation procedure as described by Atkinson in his 5 part algorithm in `Multi-view surface reconstruction using polarization` and Smith in `Height-from-Polarisation with Unknown Lighting or Albedo`, however, I was unable to complete the implementation prior to the deadline so it has not been included. 

This procedure was chosen over adding additional refinements from Kadambi because it seemed a bit easier to integrate. 