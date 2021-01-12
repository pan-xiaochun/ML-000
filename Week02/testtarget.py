from target_encoding import target_mean_v2


import numpy as np
import pandas as pd

def target_mean_v1(data:pd.DataFrame, y_name:str, x_name:str) -> np.ndarray:
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result


@profile
def main():
    y = np.random.randint(2, size=(5000, 1))
    x = np.random.randint(10, size=(5000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    result_1 = target_mean_v1(data, 'y', 'x')
    result_2 = target_mean_v2(data, 'y', 'x')

    diff = np.linalg.norm(result_1 - result_2)
    print(diff)


if __name__ == '__main__':
    main()

