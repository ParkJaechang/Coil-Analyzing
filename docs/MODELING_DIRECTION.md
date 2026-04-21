# Modeling Direction

- Primary model: `voltage waveform + frequency -> measured magnetic field waveform`
- Current is not a primary modeling parameter.
- Gain/scale is not a primary modeling parameter.
- Current, gain, and hardware values are debug or equipment reference only.
- Outputs:
  1. continuous recommended voltage waveform
  2. finite-cycle stop waveform
- Dataset strategy:
  - no large data in Git
  - use dataset root + manifest
  - cloud sync folder is acceptable
  - cloud provider API is not first priority
