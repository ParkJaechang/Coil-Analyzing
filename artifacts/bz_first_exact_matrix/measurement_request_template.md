# Measurement Request Template

- Use this template when requesting a new exact or validation measurement from the bench team.
- Fill in all required fields so the intake pipeline can classify the file without manual cleanup.
- Copy the examples exactly if the request targets the missing finite exact cell or a continuous gap-fill campaign.

## Template

- Request purpose: `<missing exact | continuous gap fill | validation | replication>`
- Upload area: `<continuous | transient | validation>`
- Waveform: `<sine | triangle>`
- Frequency Hz: `<number>`
- Cycle count: `<blank for continuous, number for transient>`
- Level: `<A for continuous, pp for transient>`
- Required columns: `Time`, `HallBz`, at least one current channel, `Voltage1` preferred
- File name example: `sinusidal/1hz_1cycle_20pp.csv`
