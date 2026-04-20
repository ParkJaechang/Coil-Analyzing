# Filename Convention

- File naming is part of the data contract because the parser uses it to recover conditions when metadata is weak.
- Optional hash prefixes are allowed and ignored for classification.
- Both `.` and `p` decimal notation are accepted in frequency and cycle tokens.
- The typo folder name `sinusidal` remains supported for backward compatibility and should not break refreshes.

## Examples

- Continuous: `<optional_hash>_<waveform>_<freq_hz>_<level_a>.csv`
- Transient: `uploads/transient/<waveform_alias>/<optional_hash>_<freq>hz_<cycle>cycle_<level>pp.csv`
- Example: `sinusidal/1hz_1cycle_20pp.csv`
- Accepted sine aliases: `sine`, `sin`, `sinusoid`, `sinusoidal`, `sinusidal`
