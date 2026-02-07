import duckdb
import os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/processed/tmp", exist_ok=True)

con = duckdb.connect()

# Ajuda o DuckDB a usar disco como “buffer”
con.execute("SET temp_directory='data/processed/tmp';")
con.execute("PRAGMA threads=4;")
con.execute("PRAGMA memory_limit='1GB';")  # ajuste pra 2GB se tiver RAM

cols = """
  event_time,
  event_type,
  product_id,
  category_id,
  category_code,
  brand,
  price,
  user_id,
  user_session
"""

# Exporta OUTUBRO
con.execute(f"""
COPY (
  SELECT {cols}
  FROM read_csv_auto('data/raw/2019-Oct.csv', header=True)
)
TO 'data/processed/events_2019_oct.parquet'
(FORMAT PARQUET);
""")

print("Saved: data/processed/events_2019_oct.parquet")

# Exporta NOVEMBRO
con.execute(f"""
COPY (
  SELECT {cols}
  FROM read_csv_auto('data/raw/2019-Nov.csv', header=True)
)
TO 'data/processed/events_2019_nov.parquet'
(FORMAT PARQUET);
""")

print("Saved: data/processed/events_2019_nov.parquet")

con.close()
