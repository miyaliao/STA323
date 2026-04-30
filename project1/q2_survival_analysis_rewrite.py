"""
Q2-(1) Survival Analysis Rewrite (single script version with detailed comments)
Reference: https://github.com/databricks-industry-solutions/survival-analysis

This script rewrites the tutorial in original words and runs in a regular PySpark + Python environment.
It outputs:
- cleaned silver dataset
- Kaplan-Meier curve and summary
- Cox PH model summary + hazard ratio plot
- Log-Logistic AFT model summary + coefficient plot
- key result tables for report
"""

import os
import warnings
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from lifelines import KaplanMeierFitter, LogLogisticAFTFitter
from lifelines.statistics import pairwise_logrank_test
from lifelines.fitters.coxph_fitter import CoxPHFitter

warnings.filterwarnings("ignore")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    # ------------------------------
    # 0) Environment setup
    # ------------------------------
    spark = SparkSession.builder.appName("Assignment3-Q2-Survival").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Write outputs under jovyan home to avoid permission issues inside container
    out_dir = "/home/jovyan/q2_outputs"
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    # ------------------------------
    # 1) Load source dataset and create silver dataset
    # ------------------------------
    # Dataset used by the official tutorial (IBM telco churn)
    data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

    # Spark may not read http URL directly in this environment, so download first then read local file
    local_csv = "/tmp/telco_customer_churn.csv"
    urllib.request.urlretrieve(data_url, local_csv)

    raw_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(local_csv)
    )

    # Convert TotalCharges to numeric where possible; invalid values become null
    raw_df = raw_df.withColumn("TotalCharges_num", F.col("TotalCharges").cast("double"))

    # Build silver table logic consistent with tutorial idea:
    # 1) churn label to binary (Yes=1, No=0)
    # 2) keep month-to-month contracts
    # 3) keep customers with internet service
    # 4) keep core fields and clean nulls on essential columns
    silver_df = (
        raw_df.withColumn(
            "churn", F.when(F.col("Churn") == "Yes", 1).when(F.col("Churn") == "No", 0).otherwise(None)
        )
        .filter(F.col("Contract") == "Month-to-month")
        .filter(F.col("InternetService") != "No")
        .withColumn("tenure", F.col("tenure").cast("double"))
        .filter(F.col("tenure").isNotNull())
        .filter(F.col("churn").isNotNull())
    )

    # Persist silver data as CSV for traceability
    silver_out_path = os.path.join(out_dir, "silver_monthly_customers.csv")
    silver_pd = silver_df.toPandas()
    silver_pd.to_csv(silver_out_path, index=False, encoding="utf-8-sig")

    # ------------------------------
    # 2) Kaplan-Meier analysis
    # ------------------------------
    kmf = KaplanMeierFitter()

    T = silver_pd["tenure"].astype(float)
    E = silver_pd["churn"].astype(float)

    # Fit population-level survival curve
    kmf.fit(T, event_observed=E, label="Population")

    # Plot KM survival curve
    plt.figure(figsize=(8, 5))
    kmf.plot()
    plt.title("Kaplan-Meier Survival Curve (Population)")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "kaplan_meier_population.png"), dpi=160)
    plt.close()

    # Save KM summary and median survival time
    median_survival = kmf.median_survival_time_
    km_summary = kmf.survival_function_.reset_index().rename(columns={"timeline": "time"})
    km_summary.to_csv(os.path.join(out_dir, "km_survival_function.csv"), index=False, encoding="utf-8-sig")

    save_text(
        os.path.join(out_dir, "km_key_result.txt"),
        f"Kaplan-Meier median survival time: {median_survival:.4f} months\n"
    )

    # Example covariate-level log-rank test: gender
    lr_gender = pairwise_logrank_test(T, silver_pd["gender"], E)
    lr_gender.summary.to_csv(os.path.join(out_dir, "logrank_gender.csv"), encoding="utf-8-sig")

    # ------------------------------
    # 3) Cox Proportional Hazards model
    # ------------------------------
    # One-hot encode selected variables (similar to tutorial spirit)
    cox_cols = [
        "dependents", "InternetService", "OnlineBackup", "TechSupport", "PaperlessBilling"
    ]

    # Standardize column names to avoid spaces/case confusion
    work_pd = silver_pd.copy()
    work_pd = work_pd.rename(
        columns={
            "Dependents": "dependents",
            "InternetService": "internetService",
            "OnlineBackup": "onlineBackup",
            "TechSupport": "techSupport",
            "PaperlessBilling": "paperlessBilling",
        }
    )
    cox_cols = ["dependents", "internetService", "onlineBackup", "techSupport", "paperlessBilling"]

    encoded = pd.get_dummies(work_pd, columns=cox_cols, prefix=cox_cols, drop_first=False)

    # Keep a compact model frame and avoid perfect multicollinearity by selecting one level per variable
    cox_model_df = pd.DataFrame({
        "churn": encoded["churn"].astype(float),
        "tenure": encoded["tenure"].astype(float),
        "dependents_Yes": encoded.get("dependents_Yes", 0),
        "internetService_DSL": encoded.get("internetService_DSL", 0),
        "onlineBackup_Yes": encoded.get("onlineBackup_Yes", 0),
        "techSupport_Yes": encoded.get("techSupport_Yes", 0),
    }).dropna()

    cph = CoxPHFitter(alpha=0.05)
    cph.fit(cox_model_df, duration_col="tenure", event_col="churn")

    # Save summary table
    cox_summary = cph.summary.reset_index()
    cox_summary.to_csv(os.path.join(out_dir, "cox_summary.csv"), index=False, encoding="utf-8-sig")

    # Plot hazard ratios
    plt.figure(figsize=(8, 5))
    cph.plot(hazard_ratios=True)
    plt.title("Cox PH Hazard Ratios")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "cox_hazard_ratios.png"), dpi=160)
    plt.close()

    # ------------------------------
    # 4) Accelerated Failure Time model (Log-Logistic)
    # ------------------------------
    aft_cols = [
        "partner", "MultipleLines", "internetService", "OnlineSecurity", "onlineBackup", "DeviceProtection",
        "techSupport", "PaymentMethod"
    ]

    # align casing in working frame
    work_pd2 = work_pd.rename(
        columns={
            "Partner": "partner",
            "MultipleLines": "multipleLines",
            "OnlineSecurity": "onlineSecurity",
            "DeviceProtection": "deviceProtection",
            "PaymentMethod": "paymentMethod",
        }
    )

    aft_cols = [
        "partner", "multipleLines", "internetService", "onlineSecurity", "onlineBackup", "deviceProtection",
        "techSupport", "paymentMethod"
    ]

    encoded2 = pd.get_dummies(work_pd2, columns=aft_cols, prefix=aft_cols, drop_first=False)

    aft_model_df = pd.DataFrame({
        "churn": encoded2["churn"].astype(float),
        "tenure": encoded2["tenure"].astype(float),
        "partner_Yes": encoded2.get("partner_Yes", 0),
        "multipleLines_Yes": encoded2.get("multipleLines_Yes", 0),
        "internetService_DSL": encoded2.get("internetService_DSL", 0),
        "onlineSecurity_Yes": encoded2.get("onlineSecurity_Yes", 0),
        "onlineBackup_Yes": encoded2.get("onlineBackup_Yes", 0),
        "deviceProtection_Yes": encoded2.get("deviceProtection_Yes", 0),
        "techSupport_Yes": encoded2.get("techSupport_Yes", 0),
        "paymentMethod_Bank transfer (automatic)": encoded2.get("paymentMethod_Bank transfer (automatic)", 0),
        "paymentMethod_Credit card (automatic)": encoded2.get("paymentMethod_Credit card (automatic)", 0),
    }).dropna()

    aft = LogLogisticAFTFitter()
    aft.fit(aft_model_df, duration_col="tenure", event_col="churn")

    aft_summary = aft.summary.reset_index()
    aft_summary.to_csv(os.path.join(out_dir, "aft_summary.csv"), index=False, encoding="utf-8-sig")

    # Plot AFT coefficients
    plt.figure(figsize=(8, 6))
    aft.plot()
    plt.title("Log-Logistic AFT Coefficients")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "aft_coefficients.png"), dpi=160)
    plt.close()

    # ------------------------------
    # 5) Report helper text
    # ------------------------------
    report_lines = []
    report_lines.append("Q2-(1) survival analysis pipeline finished.")
    report_lines.append(f"Input rows (raw): {raw_df.count()}")
    report_lines.append(f"Input rows (silver): {len(silver_pd)}")
    report_lines.append(f"Kaplan-Meier median survival time: {median_survival:.4f}")
    report_lines.append(f"Cox concordance: {cph.concordance_index_:.4f}")
    report_lines.append(f"AFT log-likelihood: {aft.log_likelihood_:.4f}")

    save_text(os.path.join(out_dir, "q2_key_metrics.txt"), "\n".join(report_lines))

    print("Q2-(1) outputs saved to", out_dir)
    spark.stop()


if __name__ == "__main__":
    main()
