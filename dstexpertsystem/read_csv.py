from dstexpertsystem.utils.errors import Errors, is_exists_path
# import dask.dataframe as dd
import pandas as dd


def read_csv(path):
    if not is_exists_path(path):
        raise Errors.FileNotFound(path)
    df = dd.read_csv(path, index_col=0)
    return df


if __name__ == "__main__":
    from pathlib import Path
    from dstexpertsystem import PROJECT_ROOT

    ### a csv file path for testing
    path = PROJECT_ROOT/"ata"/"tests"/"test.csv"

    df = read_csv(path)
    print(df)

