# Comparison of Lowess and Random Forest


## Lowess (locally weighted scatterplot smoothing)
#### Lowess is a way of fitting a curve to data. It is a non-parametric approach because the data is not assumed to fit a type of distribution, so it can model non linear data. It uses a sliding window approach to split the data into sections. Then in each section performs a regression by minimizing least squares. The kernel function is used to determine the weight of influence a point has in relation to the focal point, and is part of the sliding window process. It can be adjusted by manipulating the kernel function. Additionally the tau hyperparameter is used to decide the span of the kernel, and therefore the number of points considered in each regression performed. A larger tau will make the regression follow the overall trend, whereas a smaller tau will make the regression more sensitive to the given data.




## Lowess Kernels

The kernels are used to define the weight of importance associated with a given point (relative to the focal point), when performing the regressions.

Tricubic

<img src="Images/tricubic.png" width="200">

Quartic

<img src="Images/quartic.png" width="200">

Epanechnikov

<img src="Images/epanechnikov.png" width="200">

----