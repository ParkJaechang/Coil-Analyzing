# Data Contract

- Uploads are accepted from four areas: continuous, transient, validation, and lcr.
- The parser can reconstruct waveform, frequency, cycle count, and level from file names when metadata is incomplete.
- At minimum, every waveform file must provide a time axis and at least one usable current/field channel.
- The exact-matrix pipeline is based on direct folder scans, so correct placement and file naming matter more than manifest edits.

## Accepted Inputs

| area | folder | required_columns | recommended_columns | filename_carries |
| --- | --- | --- | --- | --- |
| continuous | uploads/continuous | Time or Timestamp, HallBz, and at least one current channel | Voltage1, HallBx, HallBy, AmpGain, Temperature | waveform, freq_hz, current level |
| transient | uploads/transient/<waveform_alias> | Time or Timestamp, HallBz, and at least one current channel | Voltage1, HallBx, HallBy, AmpGain, Temperature, CycleNo | freq_hz, cycle_count, level_pp |
| validation | uploads/validation/<scenario> | Time or Timestamp, HallBz, and the driven current/voltage channels | AmpGain, Temperature, explicit metadata header | scenario only |
| lcr | uploads/lcr | keep the exported workbook or CSV intact | device and fixture metadata | none |
