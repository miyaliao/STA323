import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_df(df, out_csv):
    pdf = df.toPandas()
    pdf.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return pdf


def main():
    spark = SparkSession.builder.appName("Assignment3-Q1-2-3").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    input_csv = "Q1_data/diabetes_012_health_indicators_BRFSS2015.csv"
    out_dir = "q1_outputs"
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    # Load data
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(input_csv)
    )

    # ------------------------------
    # Q1-(2): Schema + distribution
    # ------------------------------
    schema_txt_path = os.path.join(out_dir, "schema.txt")
    with open(schema_txt_path, "w", encoding="utf-8") as f:
        f.write(df._jdf.schema().treeString())

    total_w = Window.partitionBy()
    class_dist_df = (
        df.groupBy("Diabetes_012")
        .count()
        .orderBy("Diabetes_012")
        .withColumn("percentage", F.round(F.col("count") / F.sum("count").over(total_w), 6))
    )
    class_dist_pdf = save_df(class_dist_df, os.path.join(out_dir, "class_distribution.csv"))

    # Indicators required by assignment
    indicators = ["HighBP", "HighChol", "BMI", "PhysActivity", "Age"]

    # DataFrame API analysis: by diabetes class, compute mean
    agg_exprs = [F.avg(c).alias(f"avg_{c}") for c in indicators]
    df_api_stats = df.groupBy("Diabetes_012").agg(*agg_exprs).orderBy("Diabetes_012")
    df_api_pdf = save_df(df_api_stats, os.path.join(out_dir, "indicator_stats_dataframe_api.csv"))

    # Spark SQL API analysis: same metrics
    df.createOrReplaceTempView("diabetes")
    sql_query = """
    SELECT
      Diabetes_012,
      AVG(HighBP) AS avg_HighBP,
      AVG(HighChol) AS avg_HighChol,
      AVG(BMI) AS avg_BMI,
      AVG(PhysActivity) AS avg_PhysActivity,
      AVG(Age) AS avg_Age
    FROM diabetes
    GROUP BY Diabetes_012
    ORDER BY Diabetes_012
    """
    sql_stats = spark.sql(sql_query)
    sql_pdf = save_df(sql_stats, os.path.join(out_dir, "indicator_stats_sql_api.csv"))

    # Compare DataFrame API vs SQL API
    cmp_pdf = df_api_pdf.merge(sql_pdf, on="Diabetes_012", suffixes=("_dfapi", "_sql"))
    for c in indicators:
        cmp_pdf[f"diff_{c}"] = cmp_pdf[f"avg_{c}_dfapi"] - cmp_pdf[f"avg_{c}_sql"]
    cmp_pdf.to_csv(os.path.join(out_dir, "indicator_stats_api_comparison.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------
    # Q1-(3): 3 observations + graphs
    # ------------------------------
    # Observation graph 1: prevalence of HighBP by class
    plt.figure(figsize=(8, 5))
    plt.bar(df_api_pdf["Diabetes_012"].astype(str), df_api_pdf["avg_HighBP"])
    plt.title("Average HighBP by Diabetes Class")
    plt.xlabel("Diabetes_012")
    plt.ylabel("Average HighBP")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "obs1_highbp_by_class.png"), dpi=160)
    plt.close()

    # Observation graph 2: BMI by class
    plt.figure(figsize=(8, 5))
    plt.plot(df_api_pdf["Diabetes_012"], df_api_pdf["avg_BMI"], marker="o")
    plt.title("Average BMI by Diabetes Class")
    plt.xlabel("Diabetes_012")
    plt.ylabel("Average BMI")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "obs2_bmi_by_class.png"), dpi=160)
    plt.close()

    # Observation graph 3: physical activity by class
    plt.figure(figsize=(8, 5))
    plt.bar(df_api_pdf["Diabetes_012"].astype(str), df_api_pdf["avg_PhysActivity"])
    plt.title("Average PhysActivity by Diabetes Class")
    plt.xlabel("Diabetes_012")
    plt.ylabel("Average PhysActivity")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "obs3_physactivity_by_class.png"), dpi=160)
    plt.close()

    # Prepare observation summary text with numeric evidence
    obs_txt = []
    row0 = df_api_pdf[df_api_pdf["Diabetes_012"] == 0.0].iloc[0]
    row1 = df_api_pdf[df_api_pdf["Diabetes_012"] == 1.0].iloc[0]
    row2 = df_api_pdf[df_api_pdf["Diabetes_012"] == 2.0].iloc[0]

    obs_txt.append("Observation 1: HighBP prevalence rises with diabetes severity.")
    obs_txt.append(
        f"Evidence: avg_HighBP class0={row0['avg_HighBP']:.4f}, class1={row1['avg_HighBP']:.4f}, class2={row2['avg_HighBP']:.4f}."
    )

    obs_txt.append("Observation 2: Average BMI is substantially higher for diabetic group (class 2).")
    obs_txt.append(
        f"Evidence: avg_BMI class0={row0['avg_BMI']:.4f}, class1={row1['avg_BMI']:.4f}, class2={row2['avg_BMI']:.4f}."
    )

    obs_txt.append("Observation 3: Physical activity tends to be lower among higher-risk classes.")
    obs_txt.append(
        f"Evidence: avg_PhysActivity class0={row0['avg_PhysActivity']:.4f}, class1={row1['avg_PhysActivity']:.4f}, class2={row2['avg_PhysActivity']:.4f}."
    )

    with open(os.path.join(out_dir, "q1_observations.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(obs_txt))

    # Save first-5-lines outputs for report screenshot alternative
    with open(os.path.join(out_dir, "first5_preview.txt"), "w", encoding="utf-8") as f:
        f.write("Class distribution (first rows):\n")
        f.write(class_dist_pdf.head().to_string(index=False))
        f.write("\n\nDataFrame API indicator stats:\n")
        f.write(df_api_pdf.head().to_string(index=False))
        f.write("\n\nSQL API indicator stats:\n")
        f.write(sql_pdf.head().to_string(index=False))

    print("Done. Outputs saved to:", out_dir)
    spark.stop()


if __name__ == "__main__":
    main()
