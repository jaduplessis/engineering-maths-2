from scipy import stats
import math
import re
import numpy as np
from random import choices
from scipy.stats import chisquare, chi2_contingency, chi2, poisson


def p_value_to_z_score(p_value):
    z = stats.norm.ppf(p_value)
    return z


def p_value_to_t_score(p_value, sample_size):
    t = stats.t.ppf(p_value, sample_size)
    return t


def z_score_to_p_value(z_score):
    p = stats.norm.cdf(z_score)
    return p


def t_score_to_p_value(t_score, sample_size):
    p = stats.t.cdf(t_score, sample_size)
    return p


def mean(args):
    val_sum = sum(args)
    return val_sum / len(args)


def median(*args):
    args = sorted(args)
    if len(args) % 2 == 0:
        i = round((len(args) + 1) / 2)
        j = i - 1
        return (args[i] + args[j]) / 2
    else:
        k = round(len(args) / 2)
        return args[k]


def variance(args):
    mean_val = mean(args)
    numerator = 0
    for i in args:
        numerator += (i - mean_val) ** 2
    denominator = len(args) - 1
    return numerator / denominator


def standard_deviation(args):
    return math.sqrt(variance(args))


def coefficient_variation(*args):
    return standard_deviation(*args) / mean(*args)


def covariance(list1, list2):
    mean_1 = mean(list1)
    mean_2 = mean(list2)
    numerator = 0
    for i in range(len(list1)):
        numerator += (list1[i] - mean_1) * (list2[i] - mean_2)
    denominator = len(list1) - 1
    return numerator / denominator


def correlation_coefficient(list_1, list_2):
    s1 = standard_deviation(list_1)
    s2 = standard_deviation(list_2)
    return covariance(list_1, list_2) / (s1 * s2)


def fisher_transformation(correlation_coeff, pearsons_coeff, significance, sample_size):
    r = correlation_coeff
    rho = pearsons_coeff
    statistic = 0.5 * (math.log((1+r) / (1-r)) - math.log((1 + rho) / (1 - rho))) * math.sqrt(sample_size - 3)
    z_score = p_value_to_z_score(1-significance/2)
    if abs(statistic) < abs(z_score):
        print("Cannot reject null hypothesis")
    else:
        print("Reject null hypothesis")
    return statistic, z_score


def normalize_list(*args):
    sd_list = standard_deviation(*args)
    return [(i - mean(*args)) / sd_list for i in args]


def sample_error(*args):
    sd_list = standard_deviation(*args)
    return sd_list / math.sqrt(len(args))


def get_confidence_interval_normal(sample_mean, confidence, sd, sample_size):
    alpha_val = 1 - confidence
    critical_probability = 1 - alpha_val / 2
    z_code = p_value_to_z_score(critical_probability)
    print("Z Code: {:.3f}".format(z_code))
    x = sample_mean - (z_code * (sd / math.sqrt(sample_size)))
    y = sample_mean + (z_code * (sd / math.sqrt(sample_size)))
    print("Confidence Interval:")
    print("Low value: {:.2f}".format(x))
    print("High value: {:.2f}".format(y))


def get_confidence_interval_t(sample_mean, confidence, sd, sample_size):
    alpha_val = 1 - confidence
    critical_probability = 1 - alpha_val / 2
    t_code = p_value_to_t_score(critical_probability, sample_size - 1)
    print("T Code: {:.3f}".format(t_code))
    x = sample_mean - (t_code * (sd / math.sqrt(sample_size)))
    y = sample_mean + (t_code * (sd / math.sqrt(sample_size)))
    print("Confidence Interval:")
    print("Low value: {:.2f}".format(x))
    print("High value: {:.2f}".format(y))


def get_z_score(sample, population_mean):
    sample_mean = mean(*sample)
    sd = standard_deviation(*sample)
    sample_size = len(sample)
    return (sample_mean - population_mean) / (sd / math.sqrt(sample_size))


def get_t_score(sample, population_mean):
    sample_mean = mean(*sample)
    sample_deviation = standard_deviation(*sample)
    sample_size = len(sample)
    return(sample_mean - population_mean) / (sample_deviation / math.sqrt(sample_size))


def t_critical_value(significance_value, dof):
    critical_value = abs(stats.t.ppf(significance_value, dof))
    return critical_value


def bootstrap_confidence_interval(sample, significance):
    xbar = [mean(choices(sample, k=10)) for i in range(0, 10000)]
    xbar.sort()
    divider = int(200/significance)
    confidence_interval = [xbar[10_000//divider], xbar[(10_000*(divider-1))//divider]]
    return confidence_interval


def chisquared_test(observed, expected, significance):
    x = chisquare(observed, f_exp=expected)
    sum = x[0]
    pvalue = x[1]
    if x[1] < significance:
        print("Reject null hypothesis")
    else:
        print("Accept null hypothesis")
    return sum, pvalue


def chi2_critical_value(confidence, dof):
    chi2_critical = chi2.ppf(confidence, dof)
    return chi2_critical


def independence_test(significance_value, data): #data takes two sets of values to test between.
    stat, p, dof, expected = chi2_contingency(data)
    # interpret p-value
    print("p value is " + str(p))
    if p <= significance_value:
        print('Dependent (reject H0)')
    else:
        print('Independent (H0 holds true)')
    return stat, dof


def xbar(sample):
    total = 0
    Total = 0
    for i in range(len(sample)):
        total += sample[i] * i
        Total += sample[i]
    mu = total / Total
    return Total, mu


def poisson_probability(sample):
    mu = xbar(sample)[1]
    total = xbar(sample)[0]
    expected = []
    for i in range(len(sample)):
        p = math.exp(-mu) * (mu ** i) / math.factorial(i)
        expected.append(round(p * total, 2))
    return expected


def goodness_of_fit(observed, expected, significance, dof):
    confidence = 1 - significance
    critical_value = chi2_critical_value(confidence, dof)
    statistic, pvalue = chisquared_test(observed, expected, significance)
    return statistic, critical_value
