import numpy as np
from relazioni import associations

v1, v2 = np.array([1, 1, 2]), np.array([1, 1, 2])

matth_corr = associations.matthews_corr(v1, v2)
print(matth_corr) # 1.0

v1, v2 = np.array([1, 1, 2]), np.array([2, 1, 2])

matth_corr = associations.matthews_corr(v1, v2)
print(matth_corr) # 0.5