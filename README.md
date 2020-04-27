## Issue

Distance-based classifiers do not work well in the presence of high-class imbalance.

## Hypothesis

We can improve it by putting more weight on training samples belonging to the rare class.

## Weighting

We wanted a solution that doesn't need parameter tuning. And we wanted to support multi-class problems. Traditionally, multi-class problems can be decomposed into two-class problems two-ways:

* one-vs-one (OVO),
* one-vs-rest (OVS).

In **OVS**, the training sample weights were adjusted by:

```
w_c = (#s-#s_c)/#s
```

where `#s` is the count of training samples and `#s_c` is the count of training samples belonging to the class `c`.

In **OVO**, the training sample weights were adjusted with:

```
w_1*p_1 = w_2*p_2 = w_3*p_3 = ... = w_C*p_C
```

where `p` is a probability of observing given class in the training set and `C` is the count of unique classes.

To make the solution unique, we added constraint:

```
w_1+w_2+w_3+...+w_C = 1
```

## Classifier

We used Inverted Indexes of Neighbors Classifier (IINC). IINC is a variant of k-NN, where the nearest neighbors are weighted. But contrary to the common weighting schema, the weight `w_i` is not: 

```
w_i = 1/d_i
```

where `d_i` is the distance of the scored sample to the `i`-th sample, but rather:

```
w_i = 1/i
```

where `i` is the index from the nearest neighbor (starting with 1) to the farthest neighbor over all the training data (there is no tunable parameter - we always use all the training samples).

## Data sets

We used OpenML repository. The selection criteria for data sets:

* multi-class classification problem,
* the biggest class imbalance is at least 2:1, 
* only numerical features (for simplification),
* at most 3000 samples (to keep runtime low),
* at most 30 features (to keep runtime low),
* dense features (sparse data are a different beast).

## Measures

We used 3 different classification measures:

1. Cohen’s kappa (as a representant of thresholding measures),
2. AUC-ROC in OVO variant (as a representant of ranking measures),
3. Brier score (as a representative of calibration measures).

While by no means exhaustive, it covers the three basic types of classification measures.

## Protocol

We included results from 3-NN for comparison.

## Results

1. IINC is on average better than untuned k-NN (3-NN). 
2. IINC-OVR is better than IINC-OVO, but it fails to beat plain IINC when it comes to calibration.