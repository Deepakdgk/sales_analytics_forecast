from flask import Flask, render_template, request, send_from_directory
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import base64
import os

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# ---------------- CONFIG ----------------
REQUIRED_COLUMNS = {
    "date", "month", "area", "product",
    "sale_count", "sale_amount", "gst", "net_value", "profit"
}

REPORT_DIR = "static/report"
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------- HELPERS ----------------
def validate_data(df):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def generate_pdf(summary, area_sales, forecast_data, chart_path):
    pdf_path = f"{REPORT_DIR}/sales_analysis_report.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Sales Analytics & Forecast Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Summary
    elements.append(Paragraph("<b>Business Summary</b>", styles["Heading2"]))
    summary_table = Table([
        ["Total Sales", f"₹ {summary['total_sales']:.2f}"],
        ["Total Profit", f"₹ {summary['total_profit']:.2f}"],
        ["Total Units Sold", summary["total_units"]],
        ["Total GST", f"₹ {summary['total_gst']:.2f}"]
    ])
    summary_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 15))

    # Area Sales
    elements.append(Paragraph("<b>Area Wise Sales</b>", styles["Heading2"]))
    area_table = Table(
        [["Area", "Sales"]] +
        [[a["area"], f"₹ {a['sale_amount']:.2f}"] for a in area_sales]
    )
    area_table.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 1, colors.black)]))
    elements.append(area_table)
    elements.append(Spacer(1, 15))

    # Forecast Table
    elements.append(Paragraph("<b>30-Day Sales Forecast</b>", styles["Heading2"]))
    forecast_table = Table(
        [["Date", "Forecast Sales"]] +
        [[d, f"₹ {v:.2f}"] for d, v in forecast_data]
    )
    forecast_table.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 1, colors.black)]))
    elements.append(forecast_table)
    elements.append(Spacer(1, 15))

    # Chart
    elements.append(Paragraph("<b>Forecast Trend</b>", styles["Heading2"]))
    elements.append(Image(chart_path, width=400, height=200))

    doc.build(elements)
    return pdf_path

# ---------------- ROUTES ----------------
@app.route("/")
def upload_page():
    return render_template("upload.html")

@app.route("/dashboard", methods=["POST"])
def dashboard():
    file = request.files["file"]
    df = pd.read_excel(file)

    validate_data(df)

    summary = {
        "total_sales": float(df["sale_amount"].sum()),
        "total_profit": float(df["profit"].sum()),
        "total_units": int(df["sale_count"].sum()),
        "total_gst": float(df["gst"].sum())
    }

    area_sales = df.groupby("area", as_index=False)["sale_amount"].sum()
    month_sales = df.groupby("month", as_index=False)["sale_amount"].sum()
    product_sales = df.groupby("product", as_index=False)["sale_amount"].sum()

    return render_template(
        "dashboard.html",
        summary=summary,
        area_sales=area_sales.to_dict("records"),
        month_sales=month_sales.to_dict("records"),
        product_sales=product_sales.to_dict("records"),
        data=df.to_json(date_format="iso", orient="records")
    )

@app.route("/forecast", methods=["POST"])
def forecast():
    df = pd.read_json(request.form["data"])
    df["date"] = pd.to_datetime(df["date"])

    daily_sales = df.groupby("date", as_index=False)["sale_amount"].sum()
    daily_sales = daily_sales.sort_values("date")
    daily_sales["day"] = np.arange(len(daily_sales))

    # Model
    model = LinearRegression()
    model.fit(daily_sales[["day"]], daily_sales["sale_amount"])

    # Forecast
    future_days = 30
    future_X = np.arange(len(daily_sales), len(daily_sales) + future_days).reshape(-1, 1)
    forecast_values = model.predict(future_X)

    forecast_dates = pd.date_range(
        daily_sales["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=future_days
    )

    colors = [
        "green" if i == 0 or forecast_values[i] >= forecast_values[i-1] else "red"
        for i in range(len(forecast_values))
    ]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(daily_sales["date"], daily_sales["sale_amount"], label="Historical", color="blue")

    for i in range(len(forecast_values) - 1):
        plt.plot(
            forecast_dates[i:i+2],
            forecast_values[i:i+2],
            color=colors[i],
            linewidth=2
        )

    plt.scatter(forecast_dates, forecast_values, color=colors)
    plt.title("30-Day Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales Amount (₹)")
    plt.grid(True)
    plt.xticks(rotation=45)

    chart_path = f"{REPORT_DIR}/forecast_chart.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    with open(chart_path, "rb") as f:
        plot_url = base64.b64encode(f.read()).decode()

    forecast_data = list(zip(
        forecast_dates.strftime("%Y-%m-%d"),
        [float(v) for v in forecast_values]
    ))

    generate_pdf(
        summary={
            "total_sales": float(df["sale_amount"].sum()),
            "total_profit": float(df["profit"].sum()),
            "total_units": int(df["sale_count"].sum()),
            "total_gst": float(df["gst"].sum())
        },
        area_sales=df.groupby("area", as_index=False)["sale_amount"].sum().to_dict("records"),
        forecast_data=forecast_data,
        chart_path=chart_path
    )

    return render_template(
        "forecast.html",
        plot_url=plot_url,
        forecast_data=forecast_data,
        pdf_available=True
    )

@app.route("/download-pdf")
def download_pdf():
    return send_from_directory(REPORT_DIR, "sales_analysis_report.pdf", as_attachment=True)

@app.route("/download-template")
def download_template():
    return send_from_directory(
        directory="static/template",
        path="Sample_template.xlsx",
        as_attachment=True
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



