### Start this project

Run the below code before this project

conda d100_env create
conda activate d100_env

pre-commit install
pip install --no-build-isolation -e .

### How to proceed this project

eda_cleaning.ipynb: This is an EDA note book explaing the overall dataset.

Code for preprocessing and creating the dataset:
python -m data.prepare_data

Code for modeling:
python -m modeling.model_training
python -m modeling.model_tuning

Code for testing my transformer:
pytest

Code for evaluating:
python -m evaluating.evaluating_model
python -m evaluating.compare_model

Check the result of evaluation:
In the reports\figures folder, you can find the “predicted vs. actual” plots and the relevant features.

### Geographic aggregation of company locations

- Asia: China, India, Japan, South Korea, Israel
- Europe: Germany, United Kingdom, Austria, Switzerland, Norway, Finland, Ireland
- US: United States
- Australia: Australia
- Other: All remaining countries

# Aggregation of industry

- Tech & Telecom：Technology, Telecommunications
- Finance & Real Estate：Finance, Real Estate
- Public & Social：Healthcare, Education, Government
- Manufacturing：Manufacturing, Automotive
- Energy：Energy
- Consumer & Transport：Transportation, Retail
- Media & Entertainment：Media, Gaming
- Consulting：Consulting
