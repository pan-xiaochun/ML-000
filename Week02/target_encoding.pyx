import numpy as np
cimport numpy as cnp
import pandas as pd


def target_mean_v2(data:pd.DataFrame, y_name:str, x_name:str) -> np.ndarray:
  cdef:
    int datarows = data.shape[0]
    cnp.ndarray[cnp.float64_t] result = np.zeros(datarows, dtype=np.float64)
    dict value_dict = {}
    dict count_dict = {}
    cnp.ndarray[cnp.int_t] x_column = data[x_name].values
    cnp.ndarray[cnp.int_t] y_column = data[y_name].values

  for i in range(datarows):
    i_xcol = x_column[i]
    i_ycol = y_column[i]
    if i_xcol not in value_dict:
      value_dict[i_xcol] = i_ycol
      count_dict[i_xcol] = 1
    else:
      value_dict[i_xcol] += i_ycol
      count_dict[i_xcol] += 1

  for i in range(datarows):
    result[i] = (value_dict[x_column[i]] - y_column[i]) / (count_dict[x_column[i]] - 1)

  return result
