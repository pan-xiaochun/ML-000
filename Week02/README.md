- 编译方法
python setup.py install
kernprof -l -v testtarget.py

- 运行结果
(base) pxc@iZuf60a8w2ct7ts4lyss39Z:~/ml/ML-000/Week02/temp$ kernprof -l -v testtarget.py
0.0
Wrote profile results to testtarget.py.lprof
Timer unit: 1e-06 s

Total time: 0.900019 s
File: testtarget.py
Function: main at line 23

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    23                                           @profile
    24                                           def main():
    25         1        138.0    138.0      0.0      y = np.random.randint(2, size=(5000, 1))
    26         1        137.0    137.0      0.0      x = np.random.randint(10, size=(5000, 1))
    27         1        951.0    951.0      0.1      data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    28         1     895949.0 895949.0     99.5      result_1 = target_mean_v1(data, 'y', 'x')
    29         1       1257.0   1257.0      0.1      result_2 = target_mean_v2(data, 'y', 'x')
    30
    31         1       1535.0   1535.0      0.2      diff = np.linalg.norm(result_1 - result_2)
    32         1         52.0     52.0      0.0      print(diff)

