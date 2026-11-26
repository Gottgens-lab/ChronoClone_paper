# Locked deterministic reference

`deterministic_profiles.csv` contains 12 rows and 52 columns. Each row is one
deterministic simulation program from the neighbouring
`in_silico/simulation_generate.ipynb`, serialized with `%.17g`.

Columns are time-major:

```text
t1:  HSC, HSPC, MEP, CMP, Mk_prog, MEMP, GMP, GMDP, Mast_prog,
       Baso_prog, Neu_prog, DC_prog, Mono_prog
t2:  same 13-state order
t3:  same 13-state order
t4:  same 13-state order
```

The file SHA-256 is:

```text
e4275d94c1513026dc684bcca44703135bba4af4694ec55cd714d3c0fa21bf6f
```

`program_manifest.csv` gives the 12 clone counts, totalling 517. Its truth
labels are simulation-only labels and are not encoded in clone IDs or passed
to native partition fitting.

The profiles preserve the source notebook's deterministic rounding and
scaling behavior, including the late zero totals in its dying-out program.
They do not use the source notebook's already 20%-noised exported matrices.

To verify the checkpoint from the trusted source notebook:

```bash
python scripts/extract_deterministic_profiles.py --check-only
```
