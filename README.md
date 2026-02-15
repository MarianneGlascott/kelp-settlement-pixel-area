Kelp Settlement Pixel-Area Quantification

This repository provides the Python implementation of the image-based settlement quantification workflow described in Supplementary Material S11 of:

Glascott et al. (2026) — Zoospore motility governs encounter potential and settlement outcomes in habitat-forming kelps.

Purpose

The script quantifies early gametophyte establishment on ceramic tiles following a 24-hour settlement period. Rather than attempting to count individual gametophytes, the workflow generates a binary mask of attached material and calculates total non-zero pixels as a proxy for surface coverage.

This pixel-area metric:

Integrates both number and size of attached individuals

Is robust to clustering and overlap

Avoids observer bias

Enables reproducible batch processing

Method Summary

The workflow:

Converts images to grayscale

Applies black-hat morphological filtering

Uses fixed global thresholding

Filters connected components by size

Computes total pixel area

Optionally generates quality-control overlays

All parameters were fixed and applied uniformly across species, seasons, and treatments.

This script is provided to ensure computational transparency and reproducibility of the settlement metric used in the manuscript. All threshold values and morphological parameters were defined prior to batch processing and were not tuned on a per-image basis.

This code is released under the MIT License.