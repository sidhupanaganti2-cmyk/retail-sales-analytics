
"""
Retail Sales Analytics & Prediction
Uses Pandas and NumPy for a 7-day moving-average sales forecast.
"""

import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# 1. Project setup
# -----------------------------

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

DATA_FILE = os.path.join(DATA_DIR, "Online_Retail.xlsx")

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00352/Online%20Retail.xlsx"
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# 2. Download and load dataset
# -----------------------------

def load_data():
    if not os.path.exists(DATA_FILE):
        print("Downloading retail dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    print("Loading dataset...")
    df = pd.read_excel(DATA_FILE)

    print("Dataset loaded successfully.")
    print("Original dataset shape:", df.shape)

    return df


# -----------------------------
# 3. Clean the data
# -----------------------------

def clean_data(df):
    print("Cleaning dataset...")

    df = df.dropna(subset=["CustomerID", "Description"]).copy()

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df = df.dropna(subset=["InvoiceDate"])

    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0) &
        (~df["InvoiceNo"].astype(str).str.startswith("C"))
    ].copy()

    df["CustomerID"] = df["CustomerID"].astype(int)

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    output_file = os.path.join(
        OUTPUT_DIR, "cleaned_sales_data.csv"
    )

    df.to_csv(output_file, index=False)

    print("Cleaned dataset shape:", df.shape)
    print("Cleaned data saved to:", output_file)

    return df


# -----------------------------
# 4. Calculate business KPIs
# -----------------------------

def calculate_kpis(df):
    total_revenue = df["Revenue"].sum()
    total_orders = df["InvoiceNo"].nunique()
    total_customers = df["CustomerID"].nunique()
    total_products = df["StockCode"].nunique()

    average_order_value = (
        total_revenue / total_orders
        if total_orders else 0
    )

    print("\n========== SALES KPIs ==========")
    print(f"Total Revenue: £{total_revenue:,.2f}")
    print(f"Total Orders: {total_orders:,}")
    print(f"Total Customers: {total_customers:,}")
    print(f"Unique Products: {total_products:,}")
    print(f"Average Order Value: £{average_order_value:,.2f}")
    print("================================\n")

    kpis = pd.DataFrame({
        "Metric": [
            "Total Revenue",
            "Total Orders",
            "Total Customers",
            "Unique Products",
            "Average Order Value",
        ],
        "Value": [
            total_revenue,
            total_orders,
            total_customers,
            total_products,
            average_order_value,
        ],
    })

    kpis.to_csv(
        os.path.join(OUTPUT_DIR, "sales_kpis.csv"),
        index=False,
    )


# -----------------------------
# 5. Create visualizations
# -----------------------------

def create_visualizations(df):
    print("Creating visualizations...")

    sns.set_theme(style="whitegrid")

    # Monthly revenue
    monthly_revenue = (
        df.set_index("InvoiceDate")
        .resample("MS")["Revenue"]
        .sum()
    )

    plt.figure(figsize=(12, 6))
    monthly_revenue.plot(marker="o")
    plt.title("Monthly Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue (£)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "monthly_revenue.png")
    )
    plt.close()

    # Top 10 products by revenue
    top_products = (
        df.groupby("Description")["Revenue"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    plt.figure(figsize=(12, 7))
    top_products.plot(kind="barh")
    plt.title("Top 10 Products by Revenue")
    plt.xlabel("Revenue (£)")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "top_products.png")
    )
    plt.close()

    # Top 10 countries by revenue, excluding the United Kingdom
    country_revenue = (
        df[df["Country"] != "United Kingdom"]
        .groupby("Country")["Revenue"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    plt.figure(figsize=(12, 7))
    country_revenue.plot(kind="barh")
    plt.title("Top 10 Countries by Revenue (Excluding UK)")
    plt.xlabel("Revenue (£)")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "top_countries.png")
    )
    plt.close()

    # Revenue by day of week
    weekday_revenue = (
        df.assign(
            Weekday=df["InvoiceDate"].dt.day_name()
        )
        .groupby("Weekday")["Revenue"]
        .sum()
    )

    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    weekday_revenue = weekday_revenue.reindex(
        weekday_order
    ).dropna()

    plt.figure(figsize=(10, 5))
    weekday_revenue.plot(kind="bar")
    plt.title("Revenue by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Revenue (£)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "weekday_revenue.png")
    )
    plt.close()

    # Daily revenue distribution
    daily_revenue = (
        df.set_index("InvoiceDate")
        .resample("D")["Revenue"]
        .sum()
    )

    plt.figure(figsize=(10, 5))
    sns.histplot(daily_revenue, bins=40, kde=True)
    plt.title("Distribution of Daily Revenue")
    plt.xlabel("Daily Revenue (£)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "daily_revenue_distribution.png")
    )
    plt.close()

    print("Visualizations saved in the outputs folder.")


# -----------------------------
# 6. Forecast daily sales
# -----------------------------

def forecast_sales(df):
    print("Creating sales forecast...")

    daily_sales = (
        df.set_index("InvoiceDate")
        .resample("D")["Revenue"]
        .sum()
        .sort_index()
    )

    # Use the final 7 days as a test period when possible.
    if len(daily_sales) > 14:
        train = daily_sales.iloc[:-7]
        test = daily_sales.iloc[-7:]

        # Forecast each test day using the previous 7 days.
        history = list(train.values)
        predictions = []

        for actual in test.values:
            prediction = float(np.mean(history[-7:]))
            predictions.append(prediction)
            history.append(actual)

        mae = float(
            np.mean(
                np.abs(
                    test.values - np.array(predictions)
                )
            )
        )

        print(f"7-day test MAE: £{mae:,.2f}")

    # Forecast the next 7 days using a moving average.
    last_7_days = daily_sales.tail(7)

    average_daily_sales = float(last_7_days.mean())

    future_dates = pd.date_range(
        start=daily_sales.index.max() + pd.Timedelta(days=1),
        periods=7,
        freq="D",
    )

    forecast_values = [
        average_daily_sales
    ] * 7

    forecast = pd.DataFrame({
        "Date": future_dates,
        "ForecastRevenue": forecast_values,
    })

    forecast.to_csv(
        os.path.join(OUTPUT_DIR, "sales_forecast.csv"),
        index=False,
    )

    # Plot historical sales and forecast.
    plt.figure(figsize=(12, 6))

    daily_sales.tail(30).plot(
        label="Historical Daily Revenue"
    )

    plt.plot(
        future_dates,
        forecast_values,
        marker="o",
        linestyle="--",
        label="7-Day Moving Average Forecast",
    )

    plt.title("Daily Sales and 7-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Revenue (£)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "sales_forecast.png")
    )

    plt.close()

    print("\nNext 7 Days Sales Forecast:")
    print(forecast.to_string(index=False))
    print("\nForecast saved to outputs/sales_forecast.csv")


# -----------------------------
# 7. Main program
# -----------------------------

def main():
    try:
        df = load_data()
        df = clean_data(df)

        calculate_kpis(df)
        create_visualizations(df)
        forecast_sales(df)

        print("\nProject completed successfully!")
        print("Check the outputs folder for results.")

    except Exception as error:
        print("\nAn error occurred:", error)
        print(
            "Check your dataset, installed packages, "
            "and file paths, then try again."
        )


if __name__ == "__main__":
    main()
