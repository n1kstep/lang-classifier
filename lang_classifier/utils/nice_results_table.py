import numpy as np
import pandas as pd

from lang_classifier.constants.constants import DROP_COLS, NEW_COLS, NEW_ROWS


def nice_results_table(path: str, round_to: int = 3) -> pd.DataFrame:
    old_table = pd.read_excel(path)
    old_table.drop(columns=DROP_COLS, inplace=True)

    empty_data = np.zeros((len(NEW_ROWS), len(NEW_COLS)))
    new_table = pd.DataFrame(data=empty_data, columns=NEW_COLS)

    for i, col in enumerate(old_table.columns):
        ind_row = i // (empty_data.shape[1] - 1)
        ind_col = i % (empty_data.shape[1] - 1) + 1
        new_table.iloc[ind_row, ind_col] = old_table[col].item()

    new_table['class'] = NEW_ROWS
    new_table['support'] = new_table['support'].astype(int)

    # heuristic swap rows for nice representation
    tmp1, tmp2 = new_table.iloc[8].copy(), new_table.iloc[10].copy()
    new_table.iloc[8], new_table.iloc[10] = tmp2, tmp1

    return new_table.round(decimals = round_to)