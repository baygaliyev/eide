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
@inproceedings{aliyev2024eide,
  title     = {From GPS Traces to Individual Emission Exposure: A Data-Driven Four-Step Process},
  author    = {Aliyev, Gurban and Nanni, Mirco},
  booktitle = {International Conference on Intelligent Transport Systems (INTSYS 2024)},
  pages     = {64--82},
  year      = {2024},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-86370-7_5}
}
```

## Relation to the PhD thesis and later work

The pipeline and the paper above also formed the basis of one chapter of the
author's PhD thesis, *Analysis and Routing-Based Mitigation of Vehicular Air
Pollution Exposure* (Università di Pisa, National PhD in Artificial
Intelligence). The INTSYS paper remains the primary citation for this code.

This repository is the frozen artifact for INTSYS 2024; a newer local version
of the missing-data imputation stage (MDI) is kept separately and is not
included here. The step-2 notebook below is a cleaned derivative, not a complete
record of every MDI experiment or derived figure.

The paper also reports SARN embeddings and an SI-GCN baseline produced outside
this repository. The SARN implementation is a public fork at
[github.com/baygaliyev/SARN](https://github.com/baygaliyev/SARN). The SI-GCN
model code is a public fork of
[susurrant/flow-imputation](https://github.com/susurrant/flow-imputation) at
[github.com/baygaliyev/flow-imputation](https://github.com/baygaliyev/flow-imputation);
`si_gcn_data_prep.ipynb` here contains only the cleaned Pisa data-preparation
steps and does not train the model.

Earlier context is in Gurban Aliyev, *Analysis, Prediction and Mitigation of
Exposure to Vehicular Air Pollution Based on Multi-Source Urban Data*, SEBD
2023, CEUR Workshop Proceedings 3478, pp. 674–680,
<https://ceur-ws.org/Vol-3478/paper13.pdf>. It reports preliminary MDI/SI-GCN
experiments and discusses the PhD project that later developed into this
pipeline. The INTSYS paper remains the primary citation for this code.

## Repository contents

| File | Step | Description |
|---|---|---|
| `1_calculate_weekly_emissions.py` | 1 | Road-level emission factors for links covered by GPS traces (COPERT-style speed/acceleration functions, vehicle-type matching, map matching) |
| `util_funcs.py` | 1 | Spatial tessellation helpers used by step 1 (square tiling, selection of trajectories inside it) |
| `2_missing_emission_data_imputation.ipynb` | 2 | Model comparison for imputing emissions on road links without trajectory coverage (linear regression, XGBoost, SVR, MLP, and classification/regression variants) |
| `mdi_results.ipynb` | 2 (results) | Extracted surviving local outputs for the MDI result tables; a result-extract, not the full MDI workflow |
| `3_4_dispersion_and_exposure.ipynb` | 3–4 | Gaussian plume dispersion to concentration fields, then exposure for static and moving entities |
| `validation_of_emission_model.ipynb` | — | Rome station validation reported in the paper, plus clearly labelled supplementary sensor comparisons for Rome, Borghetto, Passi and Pisa |
| `si_gcn_data_prep.ipynb` | 2 (context) | Cleaned Pisa data preparation for the separate SI-GCN baseline; model training lives in the fork linked above |

Intended execution order: **1 → 2 → 3/4**, with
`validation_of_emission_model.ipynb` used to check the emission model.
`si_gcn_data_prep.ipynb` is a separate context artifact and is not part of that
pipeline. Validation has been consolidated into `validation_of_emission_model.ipynb`;
the Pisa/Borghetto/Passi sensor comparisons there are supplementary and were not
reported in the INTSYS paper, but are kept for provenance.

## Known limitations

Please read this before trying to run the pipeline.

**Both third-party dependencies of step 1 are resolvable.** Neither is on
PyPI, so neither appears in `requirements.txt`:

- **`mobair`** — provides trajectory filtering, speed and acceleration
  computation, map matching and emission-factor handling. It is **public**, at
  <https://github.com/matteoboh/mobility_emissions> (MIT licence), from Böhm,
  Nanni and Pappalardo, *Gross polluters and vehicle emissions reduction*,
  Nature Sustainability (2022), <https://doi.org/10.1038/s41893-022-00903-x>.
  Install it from git:

  ```bash
  pip install git+https://github.com/matteoboh/mobility_emissions.git
  ```

  That repository was formerly named `mobility_airpollution`, which is why this
  pipeline refers to it by that name in places. The old URL still redirects.
  Step 1 imports it as `mobair`, matching that project's own documented
  usage, rather than treating a repository name as a Python package.
- **`util_funcs`** — was never committed to the author's earlier repository. The
  two functions step 1 needs, `download_square_tessellation(...)` and
  `select_trajectories_within_tessellation(...)`, are provided in
  `util_funcs.py` in this repository.

**What still blocks step 1 is the input data, not the code.** Step 1 reads
`data/trajectories/<area>_trajectories_week_<n>.csv` and
`data/road_networks/<city>_network.graphml`; see *Data requirements* below. The
trajectories are restricted and are not in this repository, so the script can be
imported and its tessellation stage exercised, but not run end to end without
data obtained through the original data agreement.

The original `util_funcs` module also contained a PostgreSQL downloader for
those restricted traces, held in the KDD Lab database. That code is
deliberately **not** reproduced: it is not needed for the tessellation stage, and
`util_funcs` here reads no credentials and opens no database connection.

`mobair` pins `osmnx==1.1.1` and `scikit-mobility==1.2.2` in its own
`environment.yml`, which is consistent with the `osmnx` 1.x constraint below.

Steps 2–4 (the notebooks) depend only on publicly available packages and are
the substantive part of the pipeline. Most committed outputs are the figures
and validation results reported in the paper; the supplementary Pisa sensor
comparisons are kept for provenance.

**Environment fragility.** The notebooks target the `osmnx` 1.x API; newer 2.x
releases remove or change top-level helpers such as `osmnx.graph_from_bbox` and
`osmnx.geometries_from_bbox`, which is why `requirements.txt` pins
`osmnx>=1.4,<2.0`.

`util_funcs.select_trajectories_within_tessellation(...)` uses
`DataFrameGroupBy.apply`, which recent pandas releases warn about
(`FutureWarning: DataFrameGroupBy.apply operated on the grouping columns`). The
warning is benign here — the callback only reads `tile_ID`, which is not a
grouping column — and the code is reproduced as originally written.

## Data requirements

**No input data is included in this repository**, by design. The pipeline
requires:

- `data/trajectories/<area>_trajectories_week_<n>.csv` — vehicular GPS traces.
  Columns referenced in the code include `uid`, `lat`, `lon`, `week` and
  `week_start`; the exact time-column name depends on the
  `mobair` trajectory loader used in step 1
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
                 departure_time_start, key='YOUR_TOMTOM_API_KEY')
```

The notebook contains only the placeholder `YOUR_TOMTOM_API_KEY`; no real key
is stored in this repository. Pass your own key from your own wrapper rather
than committing one here.

TomTom is only needed to regenerate the specific alternative pedestrian routes
used in the paper's walking-exposure example. The exposure calculation itself
accepts any sequence of `(latitude, longitude)` points, so routes from
OpenStreetMap/networkx, recorded trajectories, or manually supplied coordinates
work without a TomTom key. The four routes reported in the paper were generated
with TomTom.

## License

MIT — see [LICENSE](LICENSE).

The `mobair` package required by step 1 is **not** distributed here and is not
covered by this license; it carries its own MIT license from its authors.

## Acknowledgement

This pipeline was developed with Mirco Nanni (ISTI-CNR), who co-authored the
paper and provided the emission-modelling modules. His GitHub profile is
<https://github.com/mirconanni>.
