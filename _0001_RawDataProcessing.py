###########################################################
# _0001_RETRIEVE FUNDAMENTALS
########################################################
"""
import os
import time
import shutil
from pathlib import Path
import uuid

import pandas as pd
import pyreadstat
import pyarrow as pa
import pyarrow.dataset as ds

# ----------------------------
# Paths & Config
# ----------------------------
RAW_DIR = Path("RawDatabase")
INFILE = RAW_DIR / "RawFundamentals.sas7bdat"

OUT_DIR = Path("FilteredRawData")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DS = OUT_DIR / "Fundamentals_ds"   # Parquet dataset folder

READ_CHUNK_ROWS    = int(os.environ.get("READ_CHUNK_ROWS",   "160000"))
PROGRESS_EVERY     = int(os.environ.get("PROGRESS_EVERY",    "1"))
MAX_ROWS_PER_FILE  = int(os.environ.get("MAX_ROWS_PER_FILE", "800000"))
MAX_ROWS_PER_GROUP = int(os.environ.get("MAX_ROWS_PER_GROUP","200000"))

def derive_year_from_datadate(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = s
    else:
        dt = pd.to_datetime(s, errors="coerce")
        if dt.isna().mean() > 0.5:
            s_num = pd.to_numeric(s, errors="coerce")
            dt = pd.to_datetime(s_num.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        if dt.isna().mean() > 0.5:
            s_num = pd.to_numeric(s, errors="coerce")
            dt2 = pd.to_datetime(s_num, unit="D", origin="1960-01-01", errors="coerce")
            dt = dt2 if dt2.isna().sum() < dt.isna().sum() else dt
    return dt.dt.year.astype("Int64")

def write_parquet_chunk(df_chunk: pd.DataFrame, base_dir: Path, basename_tag: str):
    if df_chunk.empty:
        return
    table = pa.Table.from_pandas(df_chunk, preserve_index=False)

    # Ensure proper partitioning (Hive-style 'year=YYYY' optional but recommended)
    part_schema = pa.schema([pa.field("year", pa.int16())])
    ds.write_dataset(
        data=table,
        base_dir=str(base_dir),
        format="parquet",
        partitioning=ds.partitioning(part_schema, flavor="hive"),
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_group=MAX_ROWS_PER_GROUP,
        max_rows_per_file=MAX_ROWS_PER_FILE,
        # <<< KEY FIX: unique basename per call >>>
        basename_template=f"part-{basename_tag}-{{i}}.parquet",
        use_threads=True,
    )

def main():
    assert INFILE.exists(), f"Input file not found: {INFILE}"

    if OUT_DS.exists():
        shutil.rmtree(OUT_DS)
    OUT_DS.mkdir(parents=True, exist_ok=True)

    total_rows = written_rows = 0
    t0 = time.time()

    reader = pyreadstat.read_file_in_chunks(
        pyreadstat.read_sas7bdat, str(INFILE), chunksize=READ_CHUNK_ROWS
    )

    for i, (df, _meta) in enumerate(reader, start=1):
        n = len(df); total_rows += n
        if "DATADATE" not in df.columns:
            raise KeyError("Column 'DATADATE' not found in the SAS file.")

        years = derive_year_from_datadate(df["DATADATE"])
        df = df.assign(year=years).dropna(subset=["year"])
        df["year"] = df["year"].astype("int16")

        # make basename unique across calls
        tag = f"{i:06d}-{uuid.uuid4().hex[:8]}"
        write_parquet_chunk(df, OUT_DS, basename_tag=tag)
        written_rows += len(df)

        if (i % PROGRESS_EVERY) == 0:
            rate = total_rows / max(1.0, time.time() - t0)
            print(
                f"[Chunk {i}] +{n:,} rows (total {total_rows:,}); "
                f"+written {len(df):,} (total written {written_rows:,}). "
                f"Throughput ~{rate:,.0f} rows/s."
            )

    print(f"[Done] Wrote {written_rows:,} / {total_rows:,} rows into {OUT_DS}. "
          f"Took {int(time.time()-t0)}s.")

if __name__ == "__main__":
    main()
"""
###########################################################
# _0002_RETRIEVE RETURNS
########################################################



import os
import time
import shutil
from pathlib import Path
import uuid

import pandas as pd
import pyreadstat
import pyarrow as pa
import pyarrow.dataset as ds

# ----------------------------
# Paths & Config
# ----------------------------
RAW_DIR = Path("RawDatabase")
INFILE = RAW_DIR / "RawReturns.sas7bdat"   # CRSP daily

OUT_DIR = Path("FilteredRawData")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DS = OUT_DIR / "Returns_ds"            # Parquet dataset folder

# Chunking (8GB-friendly; tune via env if needed)
READ_CHUNK_ROWS    = int(os.environ.get("READ_CHUNK_ROWS",   "180000"))
PROGRESS_EVERY     = int(os.environ.get("PROGRESS_EVERY",    "1"))
MAX_ROWS_PER_FILE  = int(os.environ.get("MAX_ROWS_PER_FILE", "900000"))
MAX_ROWS_PER_GROUP = int(os.environ.get("MAX_ROWS_PER_GROUP","220000"))

def derive_year_from_crsp_date(s: pd.Series) -> pd.Series:
    # 1) Already datetime?
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = s
    else:
        # 2) Try generic to_datetime (strings/ints that look like dates)
        dt = pd.to_datetime(s, errors="coerce")
        # 3) If many NAs, try yyyymmdd numeric
        if dt.isna().mean() > 0.5:
            s_num = pd.to_numeric(s, errors="coerce")
            dt_try = pd.to_datetime(s_num.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
            dt = dt_try if dt_try.isna().sum() < dt.isna().sum() else dt
        # 4) If still many NAs, assume SAS day-count since 1960-01-01
        if dt.isna().mean() > 0.5:
            s_num = pd.to_numeric(s, errors="coerce")
            dt_try = pd.to_datetime(s_num, unit="D", origin="1960-01-01", errors="coerce")
            dt = dt_try if dt_try.isna().sum() < dt.isna().sum() else dt
    return dt.dt.year.astype("Int64")

def write_parquet_chunk(df_chunk: pd.DataFrame, base_dir: Path, basename_tag: str):
    if df_chunk.empty:
        return
    table = pa.Table.from_pandas(df_chunk, preserve_index=False)

    # Partition by year (Hive layout: year=YYYY/)
    part_schema = pa.schema([pa.field("year", pa.int16())])

    ds.write_dataset(
        data=table,
        base_dir=str(base_dir),
        format="parquet",
        partitioning=ds.partitioning(part_schema, flavor="hive"),
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_group=MAX_ROWS_PER_GROUP,
        max_rows_per_file=MAX_ROWS_PER_FILE,
        basename_template=f"part-{basename_tag}-{{i}}.parquet",
        use_threads=True,
    )

def main():
    assert INFILE.exists(), f"Input file not found: {INFILE}"

    # Fresh rebuild to avoid mixing prior runs
    if OUT_DS.exists():
        shutil.rmtree(OUT_DS)
    OUT_DS.mkdir(parents=True, exist_ok=True)

    total_rows = written_rows = 0
    t0 = time.time()

    reader = pyreadstat.read_file_in_chunks(
        pyreadstat.read_sas7bdat, str(INFILE), chunksize=READ_CHUNK_ROWS
    )

    for i, (df, _meta) in enumerate(reader, start=1):
        n = len(df); total_rows += n

        if "DATE" not in df.columns:
            raise KeyError("Column 'DATE' not found in RawReturns.sas7bdat.")

        years = derive_year_from_crsp_date(df["DATE"])
        df = df.assign(year=years).dropna(subset=["year"])
        df["year"] = df["year"].astype("int16")

        tag = f"{i:06d}-{uuid.uuid4().hex[:8]}"
        write_parquet_chunk(df, OUT_DS, basename_tag=tag)
        written_rows += len(df)

        if (i % PROGRESS_EVERY) == 0:
            rate = total_rows / max(1.0, time.time() - t0)
            print(
                f"[Chunk {i}] +{n:,} rows (total {total_rows:,}); "
                f"+written {len(df):,} (total written {written_rows:,}). "
                f"Throughput ~{rate:,.0f} rows/s."
            )

    print(f"[Done] Wrote {written_rows:,} / {total_rows:,} rows into {OUT_DS}. "
          f"Took {int(time.time()-t0)}s.")

if __name__ == "__main__":
    main()

