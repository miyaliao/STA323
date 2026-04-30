from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower


def main():
    spark = SparkSession.builder.appName("Assignment2-Q2-DF-SQL").getOrCreate()

    courses_path = "Q2/Q2_data/courses.csv"
    instructors_path = "Q2/Q2_data/instructors.csv"

    courses = spark.read.option("header", True).option("inferSchema", True).csv(courses_path)
    instructors = spark.read.option("header", True).option("inferSchema", True).csv(instructors_path)

    # (1) Inner join by instructors_id
    c = courses.alias("c")
    i = instructors.alias("i")
    joined = c.join(
        i,
        c["instructors_id"] == i["id"],
        how="inner"
    )

    print("=== Q2(1) Joined Data (first 5) ===")
    joined.select(
        col("c.title").alias("course_title"),
        col("c.instructors_id"),
        col("i.display_name"),
        col("i.job_title"),
        col("c.rating").alias("course_rating"),
        col("c.created")
    ).show(5, truncate=False)

    joined_clean = joined.select(
        col("c.id").alias("course_id"),
        col("c.title").alias("course_title"),
        col("c.created").alias("created"),
        col("c.rating").alias("rating"),
        col("c.instructors_id").alias("instructors_id"),
        col("i.display_name").alias("display_name"),
        col("i.job_title").alias("job_title")
    )

    joined_clean.createOrReplaceTempView("joined_courses")
    courses.createOrReplaceTempView("courses")

    # (2) SQL: instructor with highest course rating among spark-related and created after 2018-01-01 00:00:00
    q2_2 = """
    SELECT display_name, job_title
    FROM joined_courses
    WHERE lower(course_title) LIKE '%spark%'
      AND created > '2018-01-01T00:00:00Z'
    ORDER BY CAST(rating AS DOUBLE) DESC
    LIMIT 1
    """

    print("=== Q2(2) Result ===")
    spark.sql(q2_2).show(truncate=False)

    # (3) SQL: interview/interviews, sort by rounded rating desc then created desc
    q2_3 = """
    SELECT
      id,
      title,
      ROUND(CAST(rating AS DOUBLE), 1) AS course_rating,
      created
    FROM courses
    WHERE lower(title) RLIKE 'interviews?'
    ORDER BY course_rating DESC, created DESC
    """

    print("=== Q2(3) Result (first 5) ===")
    spark.sql(q2_3).show(5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
