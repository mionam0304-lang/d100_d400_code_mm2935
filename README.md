### Start this project

conda d100_env create
conda activate d100_env

pre-commit install
pip install --no-build-isolation -e .

### How to carry out this project

python -m modeling.model_training
python -m modeling.model_tuning

### Tests

Run unit tests from the project root:

```bash
pytest -q

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
- Other：

features_columns: - "employment_type" - "company_size" - "education_required" - "years_experience" - "company_area" - "industry_group" - "num_skills"

### 作成中

model:
name: "model_1"
type: "classification"
params:
max_iter: 100
random_state: 42
```
