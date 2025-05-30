import numpy as np
import zarr
import pyarrow as pa
from pyarrow import parquet as pq
import xarray as xr
import duckdb
import time
import os
import shutil
import argparse
import dask
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
N_MOLECULES = 2000
N_REACTIONS = 20
N_TIME_POINTS = 100000  # Increased to generate more data

ZARR_STORE_PATH = "synthetic_data.zarr"
PARQUET_PATH = "synthetic_data.parquet"
CHUNK_SIZE = 1000  # Number of time points to process in each chunk


# --- Helper Functions ---
def ndarray_to_ndlist(arr: np.ndarray) -> pa.FixedSizeListArray:
    arrow_flat_array = pa.array(arr.flatten())
    nested_array = arrow_flat_array
    for dim_size in reversed(arr.shape[1:]):
        nested_array = pa.FixedSizeListArray.from_arrays(nested_array, dim_size)
    return nested_array


def generate_data_chunk(start_idx, n_molecules, n_reactions, chunk_size):
    print(
        f"Generating synthetic data chunk: {start_idx} to {start_idx + chunk_size}..."
    )
    array1 = np.random.rand(chunk_size, n_molecules, n_reactions).astype(np.float64)
    array2 = np.random.rand(chunk_size, n_molecules).astype(np.float64)
    array3 = np.random.rand(chunk_size).astype(np.float64)
    return array1, array2, array3


def get_directory_size(directory_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def save_to_zarr(array1, array2, array3, path, mode="a"):
    data = xr.Dataset(
        {
            "array1": xr.DataArray(array1, dims=("time", "mols", "rxns")),
            "array2": xr.DataArray(array2, dims=("time", "mols")),
            "array3": xr.DataArray(array3, dims=("time")),
        }
    )
    encoding = None
    append_dim = "time"
    if mode == "w":
        encoding = {
            f"array{i}": {"compressors": zarr.codecs.BloscCodec(cname="zstd", clevel=3)}
            for i in range(1, 4)
        }
        append_dim = None
    data.to_zarr(
        path, mode=mode, consolidated=False, encoding=encoding, append_dim=append_dim
    )


def save_to_parquet(array1, array2, array3, suffix=""):
    outfile = f"{PARQUET_PATH}/{suffix}.pq"
    os.makedirs(PARQUET_PATH, exist_ok=True)
    df1 = pa.table(
        {
            "array1": ndarray_to_ndlist(array1),
            "array2": ndarray_to_ndlist(array2),
            "array3": ndarray_to_ndlist(array3),
            "time": np.arange(array1.shape[0], dtype=np.int32),
        }
    )
    pq.write_table(df1, outfile, write_statistics=False, compression="zstd")


def benchmark_xarray_dask(threads=4):
    print("\nBenchmarking Xarray/Dask computation...")
    start_time = time.time()

    ds = xr.open_zarr(ZARR_STORE_PATH, consolidated=False)

    op1 = ds["array1"].mean(dim="time")
    op2 = op1 * ds["array2"].mean(dim="time")
    op3 = op2 + ds["array3"].mean(dim="time")

    with dask.config.set(pool=ThreadPoolExecutor(threads)):
        final_result_value = op3.sum().compute()

    end_time = time.time()
    print(f"  Xarray/Dask computation finished in {end_time - start_time:.2f} seconds.")
    print(f"  Xarray/Dask result (sum): {final_result_value.item()}")
    ds.close()


def benchmark_duckdb(threads=4):
    print("\nBenchmarking DuckDB computation...")
    start_time = time.time()

    con = duckdb.connect()
    con.execute("SET temp_directory = '/tmp'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET threads = {threads}")
    con.execute("SET max_memory = '4GB'")

    query = f"""
    WITH unnest_arrays AS (
        SELECT unnest(array1) AS unnested_array1, generate_subscripts(array1, 1) AS mol_idx,
            unnest(array2) AS unnested_array2, array3, time
        FROM read_parquet('{PARQUET_PATH}/*')
    ),
    unnest_second_layer AS (
        SELECT unnest(unnested_array1) AS fully_unnested_array1, mol_idx, unnested_array2, array3, time,
            generate_subscripts(unnested_array1, 1) as rxn_idx
        FROM unnest_arrays
    ),
    avg_over_time_idx AS (
        SELECT AVG(fully_unnested_array1) AS avg_array1,
            AVG(unnested_array2) AS avg_array2, AVG(array3) AS avg_array3
        FROM unnest_second_layer
        GROUP BY mol_idx, rxn_idx
    )
    SELECT SUM(avg_array1 * avg_array2 + avg_array3) FROM avg_over_time_idx
    """

    result_duckdb = con.execute(query).fetchone()

    end_time = time.time()
    print(f"  DuckDB computation finished in {end_time - start_time:.2f} seconds.")
    print(f"  DuckDB result (sum): {result_duckdb[0]}")
    con.close()


def cleanup_files():
    print("\nCleaning up created files...")
    if os.path.exists(ZARR_STORE_PATH):
        shutil.rmtree(ZARR_STORE_PATH)
        print(f"  Removed {ZARR_STORE_PATH}")
    if os.path.exists(PARQUET_PATH):
        shutil.rmtree(PARQUET_PATH)
        print(f"  Removed {PARQUET_PATH}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark synthetic data generation and processing."
    )
    parser.add_argument(
        "--mols", "-m", type=int, default=N_MOLECULES, help="Number of molecules"
    )
    parser.add_argument(
        "--rxns", "-r", type=int, default=N_REACTIONS, help="Number of reactions"
    )
    parser.add_argument(
        "--time", "-t", type=int, default=N_TIME_POINTS, help="Number of time points"
    )
    parser.add_argument(
        "--cleanup",
        "-c",
        action="store_true",
        help="Clean up generated files after benchmark",
    )
    parser.add_argument(
        "--generate",
        "-g",
        action="store_true",
        help="Generate synthetic data for benchmark",
    )
    parser.add_argument(
        "--threads",
        "-p",
        type=int,
        default=4,
        help="Number of threads to use for Dask/DuckDB",
    )
    args = parser.parse_args()

    # Generate and save data in chunks
    if args.generate:
        zarr_time = 0.0
        parquet_time = 0.0
        for start in range(0, args.time, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, args.time)
            arr1_np, arr2_np, arr3_np = generate_data_chunk(
                start, args.mols, args.rxns, end - start
            )
            zarr_start = time.time()
            save_to_zarr(
                arr1_np,
                arr2_np,
                arr3_np,
                ZARR_STORE_PATH,
                mode="a-" if start > 0 else "w",
            )
            zarr_time += time.time() - zarr_start
            parquet_start = time.time()
            save_to_parquet(arr1_np, arr2_np, arr3_np, suffix=f"chunk_{start}_{end}")
            parquet_time += time.time() - parquet_start
        print(f"Total Zarr save time: {zarr_time:.2f} seconds")
        print(f"Total Parquet save time: {parquet_time:.2f} seconds")

    parquet_size_bytes = get_directory_size(PARQUET_PATH)
    print(f"Parquet dataset size: {parquet_size_bytes / 1e6:.2f} MB")
    parquet_size_bytes = get_directory_size(ZARR_STORE_PATH)
    print(f"Zarr dataset size: {parquet_size_bytes / 1e6:.2f} MB")

    benchmark_xarray_dask(args.threads)
    benchmark_duckdb(args.threads)

    # Clean up
    if args.cleanup:
        cleanup_files()

    print("\nBenchmark complete.")
