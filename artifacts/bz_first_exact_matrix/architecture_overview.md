# Architecture Overview

- The bundle is generated from direct folder scans, validation lineage payloads, and the exact scope report.
- Exact support is defined first by measured data, then projected into LUT status, ROI ranking, and operator docs.
- Manifest files can add metadata, but they do not override direct scan visibility.

## Flow

1. report_exact_and_finite_scope.py scans uploads and defines the measured exact scope.
2. generate_bz_first_artifacts.py rebuilds exact matrix, LUT catalog, ROI, docs, and intake regression.
3. refresh_exact_supported_scope.py runs both so new uploads propagate through the bundle together.
