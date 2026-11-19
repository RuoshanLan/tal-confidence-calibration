"""Data and feature layer: everything upstream of an estimand.

Modules here may read raw predictions and annotations, match, featurize and
split. None of them may import an arm registry, a metric or the
conformal step — that direction of dependency is what keeps the prepared data
independent of the quantity it is later used to estimate.
"""
