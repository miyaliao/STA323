from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.appName("Assignment2-Q1-RDD").getOrCreate()
    sc = spark.sparkContext

    data_path = "Q1/Q1_data/departuredelays.csv"

    raw = sc.textFile(data_path)
    header = raw.first()
    rows = raw.filter(lambda x: x != header)

    # (row_id, row_text) so partitioning can be controlled by origin + pseudo-random row id hash.
    indexed = rows.zipWithIndex().map(lambda x: (x[1], x[0]))

    # key = (origin, row_id)
    pair = indexed.map(lambda x: ((x[1].split(",")[3], x[0]), x[1]))

    def custom_partitioner(key):
        origin, row_id = key
        if origin == "ATL":
            return 0
        # pseudo-random but deterministic split into partitions 1,2,3
        return (hash(row_id) % 3) + 1

    partitioned = pair.partitionBy(4, custom_partitioner).map(lambda kv: kv[1])

    # Check partition stats
    stats = partitioned.mapPartitionsWithIndex(
        lambda idx, it: [(idx, sum(1 for _ in it))]
    ).collect()

    print("=== Q1 Partition Counts ===")
    for idx, cnt in sorted(stats):
        print(f"Partition {idx}: {cnt}")

    # Validate ATL rows are all in partition 0
    atl_by_partition = pair.partitionBy(4, custom_partitioner).mapPartitionsWithIndex(
        lambda idx, it: [(idx, sum(1 for _, row in it if row.split(",")[3] == "ATL"))]
    ).collect()

    print("\n=== ATL Rows per Partition ===")
    for idx, cnt in sorted(atl_by_partition):
        print(f"Partition {idx}: {cnt}")

    print("\n=== First 5 rows after partitioning ===")
    for line in partitioned.take(5):
        print(line)

    spark.stop()


if __name__ == "__main__":
    main()
