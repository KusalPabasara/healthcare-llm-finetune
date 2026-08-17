# Dataset Sources

Provenance for everything in `data/raw/`. Fill a row per dataset **as you download it** —
the Day 12 master report needs this, and healthcare data often carries usage restrictions
that are painful to reconstruct later.

| Dataset | Source / URL | Licence | Size | Records | Downloaded | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

## Licence check

Healthcare datasets commonly restrict commercial use or redistribution. For each one, confirm
before it goes into training:

- [ ] Commercial use permitted (this is a company POC)
- [ ] Redistribution terms understood (affects sharing models on Drive)
- [ ] Attribution requirements noted
- [ ] No PHI / de-identification confirmed

## Integrity

Record a checksum so you can prove the raw data never changed:

```bash
find data/raw -type f -not -name '*.md' -not -name '.gitkeep' \
  -exec sha256sum {} \; | sort > data/raw/CHECKSUMS.txt
```

Re-run and diff at any point in the sprint to confirm the anchor held.
