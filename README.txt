BMDS2003 Data Science - Housing Prices in Malaysia
==================================================
CONTENTS
  BMDS2003_Complete_Project.ipynb    full CRISP-DM notebook (Parts 1-6)
  malaysia_house_price_data_2025.csv raw dataset
  malaysia_house_price_cleaned.csv   cleaned dataset (1,914 x 10)
  model_results.csv                  model metrics
  test_predictions.csv               hold-out predictions
  models/                            4 trained pipelines (.pkl)
  figures/                           Figures 1-22 used in the report and the app
  streamlit_app.py                   deployment prototype (light design system)
  .streamlit/config.toml             locks the app to the light theme
  assets/                            optional hero banner (.png)
  requirements.txt

TO RUN THE PROTOTYPE
  1. Keep streamlit_app.py, model_results.csv, malaysia_house_price_cleaned.csv,
     models/, figures/ and assets/ in the SAME directory.
  2. Optional: add assets/malaysia_housing_banner.png for the hero image.
  3. pip install -r requirements.txt
  4. streamlit run streamlit_app.py
