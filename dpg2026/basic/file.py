from core import as_function_node


@as_function_node("csv")
def ReadCSV(
    filename: str, header: list = [0, 1], decimal: str = ",", delimiter: str = ";"
):
    import pandas as pd

    return pd.read_csv(filename, delimiter=delimiter, header=header, decimal=decimal)
