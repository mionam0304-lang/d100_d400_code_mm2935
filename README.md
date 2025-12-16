conda d100_env create
conda activate d100_env

pre-commit install
pip install --no-build-isolation -e .

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
- Other：上記以外
