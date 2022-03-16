# Relationships
Relationships A lightweight package for strength of the relationship between variables analysis.

Documentation: https://chicodelarosa.com/relationships
Source code: https://github.com/chicodelarosa/relationships
Bug reports: https://github.com/chicodelarosa/relationships/issues

It provides easy to use functions for measuring the correlation between variables of the following natures:

### Two Continuous
A variable that can reasonably take on any value within a range. Examples of continuous variables include height, weight, exam scores, income, salary, etc.

### Two Categorical
A variable that is a category without a natural order. Examples of categorical variables are eye color, city of residence, type of dog, etc.

### At least One Ordinal
A variable with categories that have an inherent order. For instance, education level (GDE/Bachelors/Masters/PhD), income level (if grouped into high/medium/low) etc.

### One Binary and One Continuous
A variable that is a category with only two possible values. Examples of binary variables include gender (male/female) or any True/False or Yes/No variable.

Relationships currently supports 8 different association functions for investigating the relationship between variables in the following cases:

1. Two Continuous and Covariates
   * Partial Correlation (R)
2. Two Continuous and No Covariates
   * Pearson Correlation
3. Two Categorical and Two Values per Variable
   * Phi Coefficient
4. Two Categorical and More than Two Values per Variable
   * Cramer’s V
   * Theil's U
5. At Least One Ordinal
   * Kendall’s Tau
   * Spearman’s Rho
6. One Continuous and One Binary
   * Point-biserial Correlation

## Example
    >>> import numpy as np
    >>> import relationships
    
    >>> v1, v2 = np.array([1, 1, 2]), np.array([1, 1, 2])

    >>> matth_corr = relationships.matthews_corr(v1, v2)
    >>> print(matth_corr)
    1.0

    >>> v1, v2 = np.array([1, 1, 2]), np.array([2, 1, 2])

    >>> matth_corr = matthews_corr(v1, v2)
    >>> print(matth_corr)
    0.5

## Call for Contributions
The Relationships project welcomes your expertise and enthusiasm!