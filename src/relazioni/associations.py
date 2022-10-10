"""Module for the computation of several association metrics between variables."""

import numpy as np
from pandas import crosstab
from scipy.stats import (
    chi2_contingency,
    kendalltau,
    linregress,
    pointbiserialr,
    spearmanr,
)
from utils.consistency import check_variables, check_binary_categorical
from sklearn.metrics import matthews_corrcoef

def theils_u(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute Theil's U2.

    What is Theils's U?
    Theils's U is used to understand the strength of the relationship between two variables.
    To use it, your variables of interest should be categorical with two or more unique
    values per category.

    Assumptions for Theils's U
    Every statistical method has assumptions. Assumptions mean that your data must satisfy
    certain properties in order for statistical method results to be accurate.

    The assumptions for Theils's U include:
    1. Categorical variables
    For this test, your two variables must be categorical. A categorical variable is a
    variable that describes a category that doesn’t relate naturally to a number. Examples of
    categorical variables are eye color, city of residence, type of dog, etc.

    When to use the Theils's U?
    You should use the Theils's U in the following scenario:

    1. You want to know the relationship between two variables
    2. Your variables of interest are categorical
    3. You have two or more unique values per category

    (Relationship >> Two Categorical >> More than Two Values per Variable)

    Parameters
    ----------
    v1 : array_like
        A 1-D array containing multiple variables and observations.
    v2 : array_like
        A 1-D array containing multiple variables and observations.

    Returns
    -------
    out : float
       Cramér's correlation coefficient.
    """
    check_variables(v1, v2)

    out = np.sqrt(np.sum(((v1[1:] - v2[1:]) / v2[:-1]) ** 2)) / np.sqrt(
        np.sum(((v2[1:] - v2[:-1]) / v2[:-1]) ** 2)
    )

    return out


def matthews_corr(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute Matthew's correlation coefficient (phi coefficient).

    What is the Phi Coefficient or Matthew's Correlation Coefficient?
    The Phi Coefficient or Matthew's Correlation Coefficient is used to understand the
    strength of the relationship between two variables. To use it, your variables of
    interest should be binary.

    Assumptions for the Phi Coefficient or Matthew's Correlation Coefficient
    Every statistical method has assumptions. Assumptions mean that your data must satisfy
    certain properties in order for statistical method results to be accurate.

    The assumptions for the Phi Coefficient or Matthew's Correlation Coefficient include:
    1. Binary variables
    For this test, your two variables must be binary. Binary means that your variable is a
    category with only two possible values. Some good examples of binary variables include
    gender (male/female) or any True/False or Yes/No variable.

    When to use the Phi Coefficient or Matthew's Correlation Coefficient?
    You should use the Phi Coefficient or Matthew's Correlation Coefficient in the following
    scenario:

    1. You want to know the relationship between two variables
    2. Your variables of interest are binary
    3. You have only two variables

    (Relationship >> Two Categorical >> Two Values per Variable)

    Parameters
    ----------
    v1 : array_like
        A 1-D array containing multiple variables and observations.
    v2 : array_like
        A 1-D array containing multiple variables and observations.

    Returns
    -------
    out : float
        Matthews's correlation coefficient.
    """
    check_binary_categorical(v1, v2)

    out = matthews_corrcoef(v1, v2)

    return out


def cramers_v(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute Cramér's V correlation coefficient.

    What is Cramér's V?
    Cramér's V is used to understand the strength of the relationship between two variables.
    To use it, your variables of interest should be categorical with two or more unique
    values per category.

    Assumptions for Cramér's V
    Every statistical method has assumptions. Assumptions mean that your data must satisfy
    certain properties in order for statistical method results to be accurate.

    The assumptions for Cramér's V include:
    1. Categorical variables
    For this test, your two variables must be categorical. A categorical variable is a
    variable that describes a category that doesn’t relate naturally to a number. Examples of
    categorical variables are eye color, city of residence, type of dog, etc.

    When to use the Cramér's V?
    You should use the Cramér's V in the following scenario:

    1. You want to know the relationship between two variables
    2. Your variables of interest are categorical
    3. You have two or more unique values per category

    (Relationship >> Two Categorical >> More than Two Values per Variable)

    Parameters
    ----------
    v1 : array_like
        A 1-D array containing multiple variables and observations.
    v2 : array_like
        A 1-D array containing multiple variables and observations.

    Returns
    -------
    out : float
       Cramér's correlation coefficient.
    """
    check_variables(v1, v2)

    ct = crosstab(v1, v2).values

    X2 = chi2_contingency(ct, correction = False)[0]
    n = ct.sum().sum()
    dof = min(ct.shape) - 1

    out = np.sqrt(X2 / (n * dof))

    return out


def kendalls_corr(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute Kendall’s correlation coefficient.

    What is Kendall’s Tau?
    Kendall’s Tau is used to understand the strength of the relationship between two
    variables. Your variables of interest can be continuous or ordinal and should have a
    monotonic relationship.

    Assumptions for Kendall’s Tau
    Every statistical method has assumptions. Assumptions mean that your data must satisfy
    certain properties in order for statistical method results to be accurate.

    The assumptions for Kendall’s Tau include:
    1. Continuous or ordinal
    The variables that you care about must be continuous or ordinal. Continuous means that
    the variable can take on any reasonable value. Some good examples of continuous
    variables include age, weight, height, test scores, survey scores, yearly salary, etc.
    Kendall’s Tau is often used for correlation on continuous data if there are outliers in
    the data.

    Ordinal variables are categories that have an inherent order. For instance, education
    level (GDE/Bachelors/Masters/PhD), income level (if grouped into high/medium/low) etc.

    2. Monotonicity
    Your two variables should have a monotonic relationship. This means that the direction
    of the relationship between the variables is consistent. For instance, when one variable
    goes up, the other goes up (in general). In this case, a plot of the two variables would
    move consistently in the up-right direction. The relationship would also be monotonic if
    when one variable goes up, the other goes down (in general). In this case, the plot of
    the two variables would move consistently in the down-right direction.

    When to use Kendall’s Tau
    You should use Kendall’s Tau in the following scenario:

    1. You want to know the relationship between two variables
    2. Your variables of interest are continuous with outliers or ordinal
    3. You have only two variables

    (Relationship >> At Least One Ordinal)

    Parameters
    ----------
    v1 : array_like
        A 1-D array containing multiple variables and observations.
    v2 : array_like
        A 1-D array containing multiple variables and observations.

    Returns
    -------
    out : float
        Kendall's correlation coefficient.
    """
    check_variables(v1, v2)

    out = kendalltau(v1, v2).correlation

    return out


def spearmans_corr(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute Spearman's rank correlation coefficient.

    What is Spearman's Rank Correlation?
    Spearman's Rank Correlation is used to understand the strength of the relationship between two
    variables. Your variables of interest can be continuous or ordinal and should have a
    monotonic relationship.

    Assumptions for Spearman's Rank Correlation
    Every statistical method has assumptions. Assumptions mean that your data must satisfy
    certain properties in order for statistical method results to be accurate.

    The assumptions for Spearman's Rank Correlation include:
    1. Continuous or ordinal
    The variables that you care about must be continuous or ordinal. Continuous means that
    the variable can take on any reasonable value. Some good examples of continuous
    variables include age, weight, height, test scores, survey scores, yearly salary, etc.
    Spearman's Rank Correlation is often used for correlation on continuous data if there are outliers in
    the data.

    Ordinal variables are categories that have an inherent order. For instance, education
    level (GDE/Bachelors/Masters/PhD), income level (if grouped into high/medium/low) etc.

    2. Monotonicity
    Your two variables should have a monotonic relationship. This means that the direction
    of the relationship between the variables is consistent. For instance, when one variable
    goes up, the other goes up (in general). In this case, a plot of the two variables would
    move consistently in the up-right direction. The relationship would also be monotonic if
    when one variable goes up, the other goes down (in general). In this case, the plot of
    the two variables would move consistently in the down-right direction.

    When to use Spearman's Rank Correlation
    You should use Spearman's Rank Correlation in the following scenario:

    1. You want to know the relationship between two variables
    2. Your variables of interest are continuous with outliers or ordinal
    3. You have only two variables

    (Relationship >> At Least One Ordinal)

    Parameters
    ----------
    v1 : array_like
        A 1-D array containing multiple variables and observations.
    v2 : array_like
        A 1-D array containing multiple variables and observations.

    Returns
    -------
    out : float
        Spearman's rank correlation coefficient.
    """
    check_variables(v1, v2)

    out = spearmanr(v1, v2).correlation

    return out


def pointbiserial_corr(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute point-biserial correlation cofficient.

    What is Point-Biserial Correlation?
    Point-biserial correlation is used to understand the strength of the relationship
    between two variables. Your variables of interest should include one continuous and one
    binary variable.

    Assumptions for Point-Biserial correlation
    Every statistical method has assumptions. Assumptions mean that your data must satisfy
    certain properties in order for statistical method results to be accurate.

    The assumptions for Point-Biserial correlation include:
    1. Continuous and Binary
    For this test, you should have one continuous and one binary variable. Continuous means
    that the variable can take on any reasonable value. Some good examples of continuous
    variables include age, weight, height, test scores, survey scores, yearly salary, etc.

    Binary means that your variable is a category with only two possible values. Some good
    examples of binary variables include smoker(yes/no), sex(male/female) or any True/False
    or 0/1 variable.

    2. Normally Distributed
    The variable that you care about must be spread out in a normal way. In statistics, this
    is called being normally distributed (aka it must look like a bell curve when you graph
    the data). Only use Point-Biserial Correlation on your data if the variable you care about
    is normally distributed.

    3. No Outliers
    The variables that you care about must not contain outliers. Point-Biserial correlation is
    sensitive to outliers, or data points that have unusually large or small values. You can
    tell if your variables have outliers by plotting them and observing if any points are far
    from all other points.

    4. Equal Variances
    One of the assumptions of Point-Biserial correlation is that there is similar spread
    between the two groups of the binary variable. You can check for this assumption by
    plotting your continuous variable in each of your two groups and visually identifying if
    the spread of the data is similar.

    When to use Point-Biserial Correlation?
    You should use Point-Biserial Correlation in the following scenario:

    1. You want to know the relationship between two variables
    2. Your variables of interest include one continuous and one binary variable
    3. You have only two variables

    (Relationship >> One Continuous One Binary)

    Parameters
    ----------
    v1 : array_like
        A 1-D array containing multiple variables and observations.
    v2 : array_like
        A 1-D array containing multiple variables and observations.

    Returns
    -------
    out : float
        Point-Biserial correlation coefficient.

    """
    check_variables(v1, v2)

    out = pointbiserialr(v1, v2).correlation

    return out


def pearson_corr(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute Pearson's correlation coefficient.

    What is Pearson Correlation?
    Pearson Correlation is used to understand the strength of the relationship between two
    variables. Your variables of interest should be continuous, be normally distributed, be
    linearly related, and be outlier free. In addition, your variables should have a similar
    spread across their individual ranges.

    Assumptions for Pearson Correlation
    Every statistical method has assumptions. Assumptions mean that your data must satisfy
    certain properties in order for statistical method results to be accurate.

    The assumptions for Pearson Correlation include:
    1. Continuous
    The variable that you care about must be continuous. Continuous means that the
    variable can take on any reasonable value. Some good examples of continuous variables
    include age, weight, height, test scores, survey scores, yearly salary, etc.

    2. Normally Distributed
    The variable that you care about must be spread out in a normal way. In statistics,
    this is called being normally distributed (aka it must look like a bell curve when
    you graph the data). Only use an independent samples t-test with your data if the
    variable you care about is normally distributed.

    3. Linearity
    The variables that you care about must be related linearly. This means that if you
    plot the variables, you will be able to draw a straight line that fits the shape of
    the data.

    4. No Outliers
    The variables that you care about must not contain outliers. Pearson’s correlation is
    sensitive to outliers, or data points that have unusually large or small values. You
    can tell if your variables have outliers by plotting them and observing if any points
    are far from all other points.

    5. Similar Spread Across Range
    In statistics this is called homoscedasticity, or making sure the variables have a
    similar spread across their ranges.

    When to use Pearson Correlation?
    You should use Pearson Correlation in the following scenario:

    1. You want to know the relationship between two variables
    2. Your variables of interest are continuous
    3. You have no covariates

    (Relationship >> Two Continuous >> No Covariates)

    Parameters
    ----------
    v1 : array_like
        A 1-D array containing multiple variables and observations.
    v2 : array_like
        A 1-D array containing multiple variables and observations.

    Returns
    -------
    out : float
        Pearson correlation coefficient.

    """
    check_variables(v1, v2)

    out = np.corrcoef(v1, v2)[0, 1]

    return out


def partial_corr(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    Compute partial correlation coefficient.

    What is Partial Correlation?
    Partial Correlation is used to understand the strength of the relationship between
    two variables while accounting for the effects of one or more other variables.
    Your variables of interest should be continuous, be normally distributed, be linearly
    related, and be outlier free. In addition, your variables should have a similar
    spread across their individual ranges.

    Assumptions for Partial Correlation
    Every statistical method has assumptions. Assumptions mean that your data must satisfy
    certain properties in order for statistical method results to be accurate.

    The assumptions for Pearson Correlation include:
    1. Continuous
    The variable that you care about must be continuous. Continuous means that the
    variable can take on any reasonable value. Some good examples of continuous variables
    include age, weight, height, test scores, survey scores, yearly salary, etc.

    2. Normally Distributed
    The variable that you care about must be spread out in a normal way. In statistics,
    this is called being normally distributed (aka it must look like a bell curve when
    you graph the data). Only use an independent samples t-test with your data if the
    variable you care about is normally distributed.

    3. Linearity
    The variables that you care about must be related linearly. This means that if you
    plot the variables, you will be able to draw a straight line that fits the shape of
    the data.

    4. No Outliers
    The variables that you care about must not contain outliers. Pearson’s correlation is
    sensitive to outliers, or data points that have unusually large or small values. You
    can tell if your variables have outliers by plotting them and observing if any points
    are far from all other points.

    5. Similar Spread Across Range
    In statistics this is called homoscedasticity, or making sure the variables have a
    similar spread across their ranges.

    6. Covariate(s)
    You should only perform partial correlation if you have one or more covariates. A
    covariate is a variable whose effects you want to remove when examining the variable
    relationship of interest. For instance, if you’re examining the relationship between
    age and memory performance, you may be interested in removing the effects of education
    level. This way, you can be sure that education level isn’t influencing the results.

    If you have no covariates to include, you should use Pearson Correlation instead.

    When to use Partial Correlation?
    You should use Partial Correlation in the following scenario:

    1. You want to know the relationship between two variables
    2. Your variables of interest are continuous
    3. You have covariates

    (Relationship >> Two Continuous >> Covariates)

    Parameters
    ----------
    v1 : array_like
        A 1-D array containing multiple variables and observations.
    v2 : array_like
        A 1-D array containing multiple variables and observations.
    v3 : array_like
        A 1-D array containing multiple variables and observations for the covariate.

    Returns
    -------
    out : float
        R or the root means square of the coefficient of determination.
    """
    check_variables(v1, v2)
    check_variables(v2, v3)

    slope_v1_v3, intercept_v1_v3, r_v1_v3, p_v1_v3, se_v1_v3 = linregress(v3, v1)

    res_v1_v3 = v1 - (intercept_v1_v3 + slope_v1_v3 * v3)

    slope_v2_v3, intercept_v2_v3, r_v2_v3, p_v2_v3, se_v2_v3 = linregress(v3, v2)

    res_v2_v3 = v2 - (intercept_v2_v3 + slope_v2_v3 * v3)

    out = np.corrcoef(res_v1_v3, res_v2_v3)[0, 1]

    return out
