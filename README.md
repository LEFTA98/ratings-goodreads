## Ratings project simulations (for Goodreads dataset)

These experiments are meant to be run on a cluster. To run these experiments, do the following:

1. Download the goodreads reviews datasets that are relevant from [here](https://mengtingwan.github.io/data/goodreads#datasets)
2. Run `collate_raw_data.py` with the relevant paths to datasets.
3. Run `fixed_experiments.py` to save the fixed setting results to `fixed_results_df.csv`
4. Run the script in `simulate_responsive_market.py` with default argparse settings.