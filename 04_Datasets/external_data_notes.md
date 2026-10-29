# External data and large-file notes

The derived datasets in this folder are the project snapshot used by the research scripts and Shiny dashboard. The main join key is transfer_row_id and the main target is post_GA90.

## Included raw data

raw_transfer_data/archive/ contains the smaller tracked raw Transfermarkt-derived CSV files used by the feature builders, including games, player valuations, transfers, players, clubs, and related reference tables. Their contents were moved without schema or value changes.

The following raw files are present in the local working copy but are intentionally not staged because they exceed GitHub's normal 100 MB per-file limit: `appearances.csv` (approximately 142 MB), `game_events.csv` (approximately 150 MB), and `game_lineups.csv` (approximately 336 MB). A clean clone therefore requires these files through an approved external-data or Git LFS arrangement before running workflows that depend on them. They must be placed at `04_Datasets/raw_transfer_data/archive/` without changing their filenames or schemas.

## Excluded from ordinary Git staging

- raw_transfer_data/open-data-master/ is approximately 15.4 GB and is not suitable for ordinary GitHub storage.
- raw_transfer_data/excluded_archives/ contains large ZIP archives and local temporary files retained outside staged project files.

Optional performance-analysis scripts can use StatsBomb open data when it is available under raw_transfer_data/open-data-master/. The main Shiny dashboard and corrected audit use the included derived tables and do not require the entire external directory to launch.

No external download URL or licence statement is invented here. If a clean-machine reproduction needs excluded StatsBomb data, obtain it through the project's approved data source and place it at the documented path before running optional enrichment.
