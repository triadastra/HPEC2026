# Census input data

The model and dataset repositories both use the name `Celsia/HPEC2026`.
Use `repo_type="dataset"` for inputs and `repo_type="model"` for checkpoints.
`SOURCE.json` pins each repository separately and records input checksums.
The pinned dataset revision holds the corrected 30,087-series lattice; the
pre-correction 28,292-series build sits under `processed/legacy_precorrection/`
and is refused by the loader.

```bash
python scripts/release.py data
python scripts/release.py data-check
```

The bulk source archives are `PORTHS6MM` (imports) and `PORTHS6XM` (exports).
To build a new cohort from source rather than load the archived inputs:

```bash
python scripts/fetch_census_bulk.py --start 2010-01 --end 2025-12 \
  --out data/census_port/raw --ref data/census_port/reference
python scripts/build_census_lattice.py
```

The research repository's `fetch_census_ports.py` is not the lattice source;
that API/CSV helper is excluded from this Census bulk release.
Do not replace an experiment's frozen input with a rebuild mid-sweep.
