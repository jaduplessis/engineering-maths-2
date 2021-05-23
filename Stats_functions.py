from scipy import stats
import math
import re
import numpy as np
from random import choices
from scipy.stats import chisquare, chi2_contingency, chi2, poisson
import sympy as sym
x, y, z, i, j, k, t, u, v, r, a, b, c, n, L, w, T, s \
    = sym.symbols('x y z i j k t u v r a b c n L w T s')


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


def median(args):
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


def coefficient_variation(args):
    return standard_deviation(args) / mean(args)


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


def pearson_correlation_coefficient(list1, list2):
    x_mean = mean(list1)
    t_mean = mean(list2)
    num = 0
    den1 = 0
    den2 = 0
    for i in range(len(list1)):
        num += (list1[i] - x_mean)*(list2[i] - t_mean)
        den1 += (list1[i] - x_mean)**2
        den2 += (list2[i] - t_mean)**2
    r = num / (math.sqrt(den1)* math.sqrt(den2))
    return r


def fisher_approximation(pearsons_coeff):
    X = 0.5 * math.log((1+pearsons_coeff)/(1-pearsons_coeff))
    return X


def fisher_transformation(pearson_correlation_coeff, correlated, significance, sample_size):
    r = pearson_correlation_coeff
    rho = correlated
    statistic = 0.5 * (math.log((1+r) / (1-r)) - math.log((1 + rho) / (1 - rho))) * math.sqrt(sample_size - 3)
    z_score = p_value_to_z_score(1-significance/2)
    if abs(statistic) < abs(z_score):
        print("Cannot reject null hypothesis")
    else:
        print("Reject null hypothesis")
    return statistic, z_score
    # example format
    # pearson_correlation_coeff = -1.30
    # rho = 0 | this suggests null hypothesis is that they are uncorrelated
    # significance = 0.10
    # sample_size - 5

def normalize_list(args):
    sd_list = standard_deviation(args)
    return [(i - mean(args)) / sd_list for i in args]


def sample_error(args):
    sd_list = standard_deviation(args)
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


def t_critical_value(significance_value, dof, number_of_tails):
    significance_value = significance_value / number_of_tails
    critical_value = abs(stats.t.ppf(significance_value, dof))
    return critical_value


def normal_critical_value(significance_value, number_of_tails):
    significance_value = significance_value / number_of_tails
    critical_value = abs(stats.norm.ppf(significance_value))
    return critical_value


def get_t_score_from_data(sample, population_mean):
    sample_mean = mean(sample)
    sample_deviation = standard_deviation(sample)
    sample_size = len(sample)
    return(sample_mean - population_mean) / (sample_deviation / math.sqrt(sample_size))


def t_statistic(sample_mean, population_mean, sample_deviation, sample_size):
    T = (population_mean - sample_mean) / (sample_deviation / math.sqrt(sample_size))
    return T


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
    critical = chi2_critical_value(1-significance_value, dof)
    print("Critical value is {}. Test statisitic is {}. Therefore".format(critical, stat))
    if p <= significance_value:
        print('Dependent (reject H0)')
    else:
        print('Independent (H0 holds true)')
    return stat, dof


def xbar(sample_data, groupings):
    total = 0
    Total = 0
    for i in range(len(sample_data)):
        total += sample_data[i] * groupings[i]
        Total += sample_data[i]
    mu = total / Total
    return Total, mu
    # example
    # groupings = [0, 1, 2, 3, 4]
    # observed = [873, 77, 32, 16, 2]


def poisson_distribution(sample, groupings):
    mu = xbar(sample, groupings)[1]
    total = xbar(sample, groupings)[0]
    expected = []
    for i in range(len(sample)):
        p = math.exp(-mu) * (mu ** i) / math.factorial(i)
        expected.append(round(p * total, 2))
    return expected
    # example
    # observed = [873, 77, 32, 16, 2]
    # groupings = [0, 1, 2, 3, 4]


def normal_distribution(data, values_upper_bounds, mean, standard_deviation):
    expected = []
    total = sum(data)
    prior_p = 0
    length = len(values_upper_bounds)
    for i in range(length):
        z_score = (values_upper_bounds[i]-mean)/standard_deviation
        cumulative_p_value = z_score_to_p_value(z_score)
        if i == length - 1:
            p_value = 1 - prior_p
        else:
            p_value = cumulative_p_value - prior_p
        prior_p = cumulative_p_value
        expected.append(p_value*total)
    return expected
    # example format
    # observed data = [10, 32, 48, 10]
    # data_categories = [850, 900, 950, 1000]
    # s = math.sqrt(1625.3)
    # mean = 904


def goodness_of_fit(observed, expected, significance, dof):
    confidence = 1 - significance
    critical_value = chi2_critical_value(confidence, dof)
    statistic, pvalue = chisquared_test(observed, expected, significance)
    return statistic, critical_value


def statistically_significant(significance, sample1_size, sample1_mean, sample1_sd,
                              sample2_size, sample2_mean, sample2_sd, number_of_tails):
    global_deviation = math.sqrt(sample1_sd**2/sample1_size + sample2_sd**2/sample2_size)
    test_statistic = abs(sample1_mean-sample2_mean)/global_deviation
    critical_value = t_critical_value(significance, sample2_size-1, number_of_tails)
    if test_statistic < critical_value:
        print("Null hypothesis cannot be rejected")
    else:
        print("Null hypothesis can be rejected")
    print("Test statistic is: {}. Critical value is {}.".format(test_statistic, critical_value))
    # this is testing if two sets of data are significant


def residual(function, x_data, y_data):
    residuals = []
    for i in range(len(x_data)):
        residuals.append(abs(function(x_data[i], y_data[i])))
    return residuals
    # example code
    # x_data = [1, 2, 3]
    # y_data = [20, 30, 70]
    # function = lambda X, Y: 25*X - 10 - Y


def paired_t_test(data_1, data_2):
    difference = []
    for i in range(len(data_1)):
        difference.append(abs(data_1[i]-data_2[i]))
    x_bar = mean(difference)
    sd = standard_deviation(difference)
    test_stat = t_statistic(0, x_bar, sd, len(data_1))
    t_crit = t_critical_value(0.05, len(data_1)-1, 2)
    print("Test statistic is: {}. Critical value is: {}".format(test_stat, t_crit))
    if test_stat < t_crit:
        print("Accept null hypothesis. mu = 0. No difference")
    else:
        print("Reject null hypothesis. mu is different")
    # This test assumes there is no difference between the data sets
    # mu is the mean differences between the sets of values
    # data1 = [140, 190, 50, 80]
    # data2 = [145, 192, 62, 87]


def observations_needed(false_positive, false_negative, expected_mean, sd, alternative_mean, number_of_tails):
    z_score = p_value_to_z_score(false_negative)
    alpha = normal_critical_value(false_positive, number_of_tails)

    X_c = expected_mean + alpha * 20 / sym.sqrt(n)
    ans = sym.solve((X_c - 1060) / (20 / sym.sqrt(n)) - z_score)[0]
    number = math.ceil(ans)
    print('Minimum number of observations required is: {}'.format(number))
    return ans
    # example formatting
    # false_positive = 0.1  |  this is the same as the significance
    # false_negative = 0.2
    # expected_mean = 1050
    # sd = 20
    # new_mean = 1060
    # number_of_tails = 2



