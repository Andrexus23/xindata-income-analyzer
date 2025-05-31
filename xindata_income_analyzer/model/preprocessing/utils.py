import pandas as pd
from typing import List
from itertools import accumulate

def row_to_text(row: str) -> str:
    """Строку таблицы в текст."""
    return " | ".join([str(x) for x in row])


def table_to_text(table_path: str) -> str:
    """Преобразовать таблицу в текст."""
    df: pd.DataFrame = pd.read_csv(table_path)
    rows: List[str] = df.apply(row_to_text, axis=1).tolist()
    return ' | '.join(df.columns.to_list()) + '\n' + '\n'.join(rows)
    