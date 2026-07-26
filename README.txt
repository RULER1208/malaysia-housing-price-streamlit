BMDS2003 Data Science - Housing Prices in Malaysia
==================================================
CONTENTS
  BMDS2003_Complete_Project.ipynb   full CRISP-DM notebook (Parts 1-6)
  malaysia_house_price_data_2025.csv   raw dataset
  malaysia_house_price_cleaned.csv     cleaned dataset (1,914 x 10)
  model_results.csv / test_predictions.csv   evaluation outputs
  models/                            4 trained pipelines (.pkl)
  streamlit_app.py                   deployment prototype
  figures/                           all charts used in the report
  requirements.txt

TO RUN THE PROTOTYPE
  1. Keep streamlit_app.py, model_results.csv, malaysia_house_price_cleaned.csv
     and the models/ folder in the SAME directory.
  2. pip install -r requirements.txt
  3. streamlit run streamlit_app.py
