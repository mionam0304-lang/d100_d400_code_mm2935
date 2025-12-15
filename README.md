conda d100_env create
conda activate d100_env

pre-commit install
pip install --no-build-isolation -e .
