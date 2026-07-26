BMDS2003 Data Science - Housing Prices in Malaysia
==================================================
CONTENTS
  BMDS2003_Complete_Project.ipynb   full CRISP-DM notebook (Parts 1-6)
  malaysia_house_price_data_2025.csv   raw dataset
  malaysia_house_price_cleaned.csv     cleaned dataset (1,914 x 10)
  model_results.csv / test_predictions.csv   evaluation outputs
  models/                            4 trained pipelines (.pkl)
  streamlit_app.py                   deployment prototype
  assets/                            hero banner image goes here (optional)
  figures/                           all charts used in the report
  requirements.txt

TO RUN THE PROTOTYPE
  1. Keep streamlit_app.py, model_results.csv, malaysia_house_price_cleaned.csv,
     the models/ folder and the assets/ folder in the SAME directory.
  2. Optional: add assets/malaysia_housing_banner.jpg for the hero image.
     The app falls back to a gradient if it is missing.
  3. pip install -r requirements.txt
  4. streamlit run streamlit_app.py
