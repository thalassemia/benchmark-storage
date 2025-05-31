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


def ndidx_to_duckdb_expr(
    name: str, idx: list[int | list[int] | list[bool] | str]
) -> str:
    """
    Returns a DuckDB expression for a column equivalent to converting each row
    of ``name`` into an ndarray ``name_arr`` and getting ``name_arr[idx]``.
    ``idx`` can contain 1D lists of integers, boolean masks, or ``":"``
    (no 2D+ indices like ``x[[[1,2]]]``).

    .. WARNING:: DuckDB arrays are 1-indexed so this function adds 1 to every
        supplied integer index!

    Args:
        name: Name of column to recursively index
        idx: To get all elements for a dimension, supply the string ``":"``.
            Otherwise, only single integers or 1D integer lists of indices are
            allowed for each dimension. Some examples::

                [0, 1] # First row, second column
                [[0, 1], 1] # First and second row, second column
                [0, 1, ":"] # First element of axis 1, second of 2, all of 3
                # Final example differs between this function and Numpy
                # This func: 1st and 2nd of axis 1, all of 2, 1st and 2nd of 3
                # Numpy: Complicated, see Numpy docs on advanced indexing
                [[0, 1], ":", [0, 1]]

    """
    idx = idx.copy()
    idx.reverse()
    # Construct expression from inside out (deepest to shallowest axis)
    first_idx = idx.pop(0)
    if isinstance(first_idx, list):
        if isinstance(first_idx[0], int):
            one_indexed_idx = ", ".join(str(i + 1) for i in first_idx)
            select_expr = f"list_select(x_0, [{one_indexed_idx}])"
        elif isinstance(first_idx[0], bool):
            select_expr = f"list_where(x_0, {first_idx})"
        else:
            raise TypeError("Indices must be integers or boolean masks.")
    elif first_idx == ":":
        select_expr = "x_0"
    elif isinstance(first_idx, int):
        select_expr = f"x_0[{int(first_idx) + 1}]"
    else:
        raise TypeError("All indices must be lists, ints, or ':'.")
    i = -1
    for i, indices in enumerate(idx):
        if isinstance(indices, list):
            if isinstance(indices[0], int):
                one_indexed_idx = ", ".join(str(i + 1) for i in indices)
                select_expr = f"list_transform(list_select(x_{i + 1}, [{one_indexed_idx}]), x_{i} -> {select_expr})"
            elif isinstance(indices[0], bool):
                select_expr = f"list_transform(list_where(x_{i + 1}, {indices}), x_{i} -> {select_expr})"
            else:
                raise TypeError("Indices must be integers or boolean masks.")
        elif indices == ":":
            select_expr = f"list_transform(x_{i + 1}, x_{i} -> {select_expr})"
        elif isinstance(indices, int):
            select_expr = (
                f"list_transform(x_{i + 1}[{int(indices) + 1}], x_{i} -> {select_expr})"
            )
        else:
            raise TypeError("All indices must be lists, ints, or ':'.")
    select_expr = select_expr.replace(f"x_{i + 1}", name)
    return select_expr + f" AS {name}"


def generate_data_chunk(start_idx, n_molecules, n_reactions, chunk_size, onedim):
    print(
        f"Generating synthetic data chunk: {start_idx} to {start_idx + chunk_size}..."
    )
    if onedim:
        array1 = np.random.rand(chunk_size, n_molecules).astype(np.float64)
    else:
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


def save_to_zarr(array1, array2, array3, path, mode="a", onedim=False):
    if onedim:
        array1_dims = ("time", "mols")
    else:
        array1_dims = ("time", "mols", "rxns")
    data = xr.Dataset(
        {
            "array1": xr.DataArray(array1, dims=array1_dims),
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


def benchmark_xarray_dask(threads, subscript, onedim):
    print("\nBenchmarking Xarray/Dask computation...")
    start_time = time.time()

    ds = xr.open_zarr(ZARR_STORE_PATH, consolidated=False)

    if onedim:
        op1 = 1
        if subscript:
            arr1 = ds["array1"][:, :11]
            arr2 = ds["array2"][:, :11]
        else:
            arr1 = ds["array1"]
            arr2 = ds["array2"]
    else:
        if subscript:
            arr1 = ds["array1"][:, :, :11]
        else:
            arr1 = ds["array1"]
        arr2 = ds["array2"]
    op1 = arr1.mean(dim="time")
    op2 = op1 * arr2.mean(dim="time")
    op3 = op2 + ds["array3"].mean(dim="time")

    with dask.config.set(pool=ThreadPoolExecutor(threads)):
        final_result_value = op3.sum().compute()

    end_time = time.time()
    print(f"  Xarray/Dask computation finished in {end_time - start_time:.2f} seconds.")
    print(f"  Xarray/Dask result (sum): {final_result_value.item()}")
    ds.close()


def benchmark_duckdb(threads, subscript, onedim):
    print("\nBenchmarking DuckDB computation...")
    start_time = time.time()

    con = duckdb.connect()
    con.execute("SET temp_directory = '/tmp'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET max_memory = '{threads}GB'")

    if onedim:
        if subscript:
            arr1 = ndidx_to_duckdb_expr("array1", [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            arr2 = ndidx_to_duckdb_expr("array2", [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        else:
            arr1 = "array1"
            arr2 = "array2"
        query = f"""
        WITH subscript_array AS (
            SELECT {arr1}, {arr2}, array3, time
            FROM read_parquet('{PARQUET_PATH}/*')
        ),
        unnest_array AS (
            SELECT unnest(array1) AS unnested_array1,
                unnest(array2) AS unnested_array2,
                generate_subscripts(array2, 1) AS mol_idx,
                array3, time
            FROM subscript_array
        ),
        avg_over_time_idx AS (
            SELECT AVG(unnested_array1) AS avg_array1,
                AVG(unnested_array2) AS avg_array2,
                AVG(array3) AS avg_array3
            FROM unnest_array
            GROUP BY mol_idx
        )
        SELECT SUM(avg_array1 * avg_array2 + avg_array3) FROM avg_over_time_idx
        """
    else:
        if subscript:
            arr1 = ndidx_to_duckdb_expr("array1", [":", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        else:
            arr1 = "array1"
        query = f"""
        WITH subscript_array AS (
            SELECT {arr1}, array2, array3, time
            FROM read_parquet('{PARQUET_PATH}/*')
        ),
        unnest_arrays AS (
            SELECT unnest(array1) AS unnested_array1, generate_subscripts(array1, 1) AS mol_idx,
                unnest(array2) AS unnested_array2, array3, time
            FROM subscript_array
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
    parser.add_argument(
        "--subscript",
        "-s",
        action="store_true",
        help="Filter array1 to only include first 11 reactions to test subscript handling",
    )
    parser.add_argument(
        "--onedim",
        "-o",
        action="store_true",
        help="Only do calculations with at most one-dimensional arrays",
    )
    args = parser.parse_args()

    # Generate and save data in chunks
    if args.generate:
        zarr_time = 0.0
        parquet_time = 0.0
        for start in range(0, args.time, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, args.time)
            arr1_np, arr2_np, arr3_np = generate_data_chunk(
                start, args.mols, args.rxns, end - start, args.onedim
            )
            zarr_start = time.time()
            save_to_zarr(
                arr1_np,
                arr2_np,
                arr3_np,
                ZARR_STORE_PATH,
                mode="a-" if start > 0 else "w",
                onedim=args.onedim
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

    benchmark_xarray_dask(args.threads, args.subscript, args.onedim)
    benchmark_duckdb(args.threads, args.subscript, args.onedim)

    # Clean up
    if args.cleanup:
        cleanup_files()

    print("\nBenchmark complete.")
