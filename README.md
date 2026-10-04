# From GPS Traces to Individual Emission Exposure: a Data-Driven Four-Step Process

Reference implementation of the data-driven four-step pipeline for estimating
individual exposure to vehicle emissions from vehicular GPS traces.

The pipeline, as described in the paper, consists of:

1. an **emissions estimate** for roads covered by the input GPS data,
2. **emissions imputation** for the remaining road links,
3. **dispersion modelling** to distribute those emissions over the territory,
4. an **exposure estimate** for static or moving entities.

## Citation

If you use this code, please cite the associated paper:

> Gurban Aliyev and Mirco Nanni.
> *From GPS Traces to Individual Emission Exposure: A Data-Driven Four-Step Process.*
> In **International Conference on Intelligent Transport Systems (INTSYS 2024)**,
> Pisa, Italy, 5–6 December 2024. Lecture Notes of the Institute for Computer
> Sciences, Social Informatics and Telecommunications Engineering.
> Springer, pages 64–82.
> DOI: [10.1007/978-3-031-86370-7_5](https://doi.org/10.1007/978-3-031-86370-7_5)

> The conference was held in December 2024; the Springer volume was published in
> 2025. Both dates refer to the same paper.

BibTeX:

```bibtex
@inproceedings{aliyev2024gps,
  title     = {From GPS Traces to Individual Emission Exposure: A Data-Driven Four-Step Process},
  author    = {Aliyev, Gurban and Nanni, Mirco},
  booktitle = {International Conference on Intelligent Transport Systems (INTSYS 2024)},
  pages     = {64--82},
  year      = {2024},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-86370-7_5}
}
```

## Repository contents

| File | Step | Description |
|---|---|---|
| `1_calculate_weekly_emissions.py` | 1 | Road-level emission factors for links covered by GPS traces (COPERT-style speed/acceleration functions, vehicle-type matching, map matching) |
| `2_missing_emission_data_imputation.ipynb` | 2 | XGBoost regression to impute emissions for road links without trajectory coverage |
| `3_4_dispersion_and_exposure.ipynb` | 3–4 | Gaussian plume dispersion to concentration fields, then exposure for static and moving entities; also contains the validation sections |
| `validation_of_emission_model.ipynb` | — | Model validation before and after imputation, for Rome, Borghetto, Passi and Pisa |

Intended execution order: **1 → 2 → 3/4**, with
`validation_of_emission_model.ipynb` used to check the emission model.

## Known limitations

Please read this before trying to run the pipeline.

**Step 1 cannot be executed as published.** It depends on two modules that are
not available from this repository or from PyPI:

- **`util_funcs`** — `1_calculate_weekly_emissions.py` imports
  `download_square_tessellation(...)` and
  `select_trajectories_within_tessellation(...)` from a module `util_funcs`
  that has never been committed here. Without it the script fails at import.
- **`mobility_airpollution.mobair`** — provides trajectory filtering, speed and
  acceleration computation, map matching and emission-factor handling. This is
  an internal module and is not publicly released.

Steps 2–4 (the notebooks) depend only on publicly available packages and are
the substantive part of the pipeline. The notebook outputs are committed
because they are the figures and validation results reported in the paper.

**Environment fragility.** The notebooks target the `osmnx` 1.x API. A stored
traceback in `3_4_dispersion_and_exposure.ipynb` records a failure against
`osmnx` 2.x (`module 'osmnx' has no attribute 'bbox_from_place'`), which is why
`requirements.txt` pins `osmnx>=1.4,<2.0`. Three further stored tracebacks
(`KeyError: 'week'`, `NameError: name 'df_sorted' is not defined`) are
mid-notebook development artefacts, not part of the final result.

## Data requirements

**No input data is included in this repository**, by design. The pipeline
requires:

- `data/trajectories/<area>_trajectories_week_<n>.csv` — vehicular GPS traces.
  Columns referenced in the code include `uid`, `lat`, `lon`, `week` and
  `week_start`; the exact time-column name depends on the
  `mobility_airpollution` loader used in step 1
- `data/road_networks/<city>_network.graphml` — directed OSM network,
  buildable with `osmnx`
- `data/emission_functions.csv` — emission factors by speed and acceleration
- `data/modelli_auto.tar.xz` — Italian vehicle-type registry used to match
  vehicles to fuel type

Downstream tables carry per-link emissions in the columns `CO_2`, `NO_x`, `PM`
and `VOC`, keyed by `week`, `week_start`, `uid` and `road_link`.

The GPS trajectories are **restricted data** obtained under agreement and are
not redistributable. All notebooks were executed in Google Colab with the data
mounted from Google Drive. If you need these inputs you must obtain them
through the original data agreement.

## Installation

```bash
git clone https://github.com/baygaliyev/eide.git
cd eide
pip install -r requirements.txt
```

`scikit-mobility` is not reliably installable from PyPI; use conda-forge:

```bash
conda install -c conda-forge scikit-mobility rtree
```

## API keys

Step 3/4 contains `get_route_tomtom(...)`, which calls the TomTom Routing API.
The key is passed **directly as the `key` argument** of that function:

```python
get_route_tomtom(start_lat, start_lon, end_lat, end_lon,
                 departure_time_start, key=YOUR_TOMTOM_API_KEY)
```

The notebook does not read any environment variable or secrets file, and no key
is stored in this repository. Pass your own key from your own wrapper rather
than editing the notebook.

## Repository history

`3_4_dispersion_and_exposure.ipynb` (steps 3 and 4) was inadvertently truncated
to 2 bytes by a file rename in November 2024 and was non-functional until
restored from commit `5cb6b6f` (October 2026). The restored file is
byte-identical to the last good revision, verified by git blob hash
`81b7a9dd3c5a95bdbe2b20fead3bd121d492c8af`.

## License

MIT — see [LICENSE](LICENSE).

The `mobility_airpollution` package required by step 1 is **not** distributed
here and is not covered by this license.

## Acknowledgement

This pipeline was developed with Mirco Nanni (ISTI-CNR), who co-authored the
paper and provided the emission-modelling modules.
