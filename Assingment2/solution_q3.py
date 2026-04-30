import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def load_sqlite_table(spark, jdbc_url, driver, table):
    return (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table)
        .option("driver", driver)
        .load()
    )


def main():
    jar_path = os.path.abspath("Q3/Q3_data/sqlite-jdbc-3.51.3.0.jar")

    spark = (
        SparkSession.builder.appName("Assignment2-Q3-DF-SQL")
        .config("spark.jars", jar_path)
        .config("spark.driver.extraClassPath", jar_path)
        .config("spark.executor.extraClassPath", jar_path)
        .getOrCreate()
    )

    sqlite_db_path = "Q3/Q3_data/debit_card_specializing/debit_card_specializing.sqlite"
    jdbc_url = f"jdbc:sqlite:{sqlite_db_path}"
    jdbc_driver = "org.sqlite.JDBC"

    customers = load_sqlite_table(spark, jdbc_url, jdbc_driver, "customers")
    yearmonth = load_sqlite_table(spark, jdbc_url, jdbc_driver, "yearmonth")

    # --------------------------------
    # (1) Spark DataFrame API Solution
    # --------------------------------
    base = (
        yearmonth.join(customers, on="CustomerID", how="inner")
        .filter((F.col("Currency") == "CZK") & (F.col("Date").between("201301", "201312")))
        .select("CustomerID", "Segment", F.col("Consumption").cast("double").alias("Consumption"))
    )

    customer_year_total = base.groupBy("Segment", "CustomerID").agg(
        F.sum("Consumption").alias("year_total_consumption")
    )

    w = Window.partitionBy("Segment").orderBy(F.col("year_total_consumption").asc())
    segment_min_customers = customer_year_total.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1)

    annual_avg = (
        segment_min_customers.groupBy("Segment")
        .agg(
            (F.sum("year_total_consumption") / F.count("CustomerID")).alias("annual_avg_consumption")
        )
    )

    avg_map = {r["Segment"]: r["annual_avg_consumption"] for r in annual_avg.collect()}

    df_api_result = spark.createDataFrame(
        [
            ("SME-LAM", avg_map.get("SME") - avg_map.get("LAM")),
            ("LAM-KAM", avg_map.get("LAM") - avg_map.get("KAM")),
            ("KAM-SME", avg_map.get("KAM") - avg_map.get("SME")),
        ],
        ["difference_type", "difference_value"],
    )

    print("=== Q3(1) DataFrame API result ===")
    annual_avg.orderBy("Segment").show(truncate=False)
    df_api_result.show(truncate=False)

    # -------------------------
    # (2) Spark SQL API Solution
    # -------------------------
    customers.createOrReplaceTempView("customers")
    yearmonth.createOrReplaceTempView("yearmonth")

    q3_sql = """
    WITH base AS (
      SELECT y.CustomerID, c.Segment, CAST(y.Consumption AS DOUBLE) AS Consumption
      FROM yearmonth y
      JOIN customers c ON y.CustomerID = c.CustomerID
      WHERE c.Currency = 'CZK'
        AND y.Date BETWEEN '201301' AND '201312'
    ),
    customer_year_total AS (
      SELECT Segment, CustomerID, SUM(Consumption) AS year_total_consumption
      FROM base
      GROUP BY Segment, CustomerID
    ),
    ranked AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY Segment ORDER BY year_total_consumption ASC) AS rn
      FROM customer_year_total
    ),
    min_customers AS (
      SELECT Segment, CustomerID, year_total_consumption
      FROM ranked
      WHERE rn = 1
    ),
    annual_avg AS (
      SELECT Segment,
             SUM(year_total_consumption) / COUNT(CustomerID) AS annual_avg_consumption
      FROM min_customers
      GROUP BY Segment
    )
    SELECT
      MAX(CASE WHEN Segment='SME' THEN annual_avg_consumption END) -
      MAX(CASE WHEN Segment='LAM' THEN annual_avg_consumption END) AS diff_SME_LAM,
      MAX(CASE WHEN Segment='LAM' THEN annual_avg_consumption END) -
      MAX(CASE WHEN Segment='KAM' THEN annual_avg_consumption END) AS diff_LAM_KAM,
      MAX(CASE WHEN Segment='KAM' THEN annual_avg_consumption END) -
      MAX(CASE WHEN Segment='SME' THEN annual_avg_consumption END) AS diff_KAM_SME
    FROM annual_avg
    """

    print("=== Q3(2) Spark SQL result ===")
    spark.sql(q3_sql).show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
