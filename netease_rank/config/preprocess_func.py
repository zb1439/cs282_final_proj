import numpy as np


def log(x):
    return np.log(x + 1)


def identity(x):
    return x


def age_categorize(age):
    return (lambda x: 0 if x < 18 else (1 if x < 30 else (2 if x < 45 else 3)))(age)


def registered_month_categorize(month):
    return (lambda x: 0 if x < 25 else (1 if x < 50 else (2 if x < 75 else 3)))(month)


def publish_time_categorize(time):
    return (lambda x: 0 if x < 50 else (1 if x < 100 else (2 if x < 150 else 3.0)))(time)


def mlog_type_categorize(mlog):
    return int(mlog - 1)

def gender_categorize(gender):
    if np.isnan(gender):
        return 0
    return gender
