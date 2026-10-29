
# Retail Sales Analytics & Prediction

## Project Overview
This project analyzes online retail transactions to identify sales trends, top-performing products, and customer purchasing patterns. It also generates a 7-day sales forecast using a moving-average method.

## Objectives
- Clean and prepare retail transaction data.
- Calculate key sales performance indicators.
- Analyze monthly revenue and revenue by weekday.
- Identify top products and countries by revenue.
- Forecast sales for the next seven days.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- OpenPyXL

## Dataset
UCI Online Retail Dataset:
https://archive.ics.uci.edu/dataset/352/online+retail

The program downloads the dataset when it is not already present locally.

## Key Results
- Total revenue: £8,911,407.90
- Total orders: 18,532
- Total customers: 4,338
- Unique products: 3,665
- Average order value: £480.87

## Forecasting
The project uses the average revenue from the previous seven days to forecast daily revenue for the next seven days.

The seven-day test-period mean absolute error (MAE) was approximately £38,300.96.

## How to Run

1. Install Python 3.13.
2. Install the required packages:

   pip install -r requirements.txt

3. Run the project:

   python main.py

## Generated Outputs
- Sales KPI report
- Monthly revenue chart
- Top products chart
- Top countries chart
- Weekday revenue chart
- Daily revenue distribution chart
- Seven-day sales forecast CSV and chart

## Author
Panuganti sidhu
