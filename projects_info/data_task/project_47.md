# Spatial covariance and Landau-Ginzburg models for embedded empirical networks 

Score: 1.2

*Statistical field theory provides a powerful framework for studying the emergence of long-range correlations in spatially extended systems. Applying these ideas to network science is challenging, but spatially embedded networks offer a natural starting point: their topology can be compared with covariance structures induced by statistical fields defined over space. This project develops a computational proof of concept connecting graph kernels, spatial covariance, and Landau-Ginzburg field theories on empirical spatial networks.*

--

Students should start from spatially embedded empirical networks, initially using the GridKit dataset:

https://zenodo.org/records/47317

The dataset contains European and North-American high-voltage power-grid extracts, with stations provided as vertices and power lines provided as links. Other spatial networks may be considered as optional extensions if they include node coordinates and an edge list.

The objective is to describe the topology of a spatially embedded empirical network through spatial covariance kernels and to test whether these kernels can be reproduced by a simple Landau-Ginzburg theory. The project should first establish a direct graph-kernel baseline and then infer the coefficients of homogeneous and non-homogeneous field-theoretic models whose covariance structure approximates the empirical network kernel.

The main computational task is to compare three levels of description:

1. empirical graph kernels fitted directly from the observed network;
2. homogeneous Landau-Ginzburg covariance models with an even potential up to quartic order;
3. non-homogeneous Landau-Ginzburg covariance models in which the potential coefficients vary in space.

Expected output:

- cleaned node and edge files for at least one spatial empirical network;
- node coordinates projected into a suitable metric coordinate system;
- empirical graph kernel or adjacency-probability kernel as a function of spatial distance;
- fitted homogeneous and non-homogeneous kernel models;
- inferred Landau-Ginzburg potential coefficients;
- plots comparing empirical kernels and model-reconstructed covariance functions;
- a short report explaining the theoretical assumptions, numerical procedure, limitations, and results.

The analysis should include:

1. reconstruction of the empirical spatial network from node and edge files;
2. computation of pairwise spatial distances between nodes;
3. estimation of a homogeneous graph kernel, for example the probability of connection as a function of distance;
4. estimation of a non-homogeneous graph kernel, for example a distance kernel with spatially varying local density or node-dependent propensity terms;
5. definition of a statistically homogeneous Landau-Ginzburg model with an even potential up to quartic terms;
6. numerical inference of the homogeneous potential coefficients so that the field covariance approximates the fitted homogeneous graph kernel;
7. extension to a non-homogeneous potential whose coefficients vary over space or over coarse spatial regions;
8. inference of the non-homogeneous coefficients so that the resulting covariance approximates the fitted non-homogeneous graph kernel;
9. quantitative comparison between empirical kernels and field-theoretic covariance models using error measures such as mean squared error or likelihood-based scores;
10. discussion of whether the field-theoretic description captures nontrivial spatial organization beyond a simple distance-decay kernel.

The homogeneous graph-kernel baseline may be defined as:

- bin node pairs by spatial distance;
- compute the empirical probability that two nodes at distance r are connected;
- fit a parametric kernel, such as exponential, Gaussian, power-law cutoff, or stretched exponential decay.

The non-homogeneous graph-kernel baseline may include:

- spatially varying node density;
- node-specific or region-specific fitness parameters;
- local corrections to the distance-decay kernel;
- coarse-grained geographic regions with different kernel parameters.

The Landau-Ginzburg component should be implemented at a simplified numerical level. Students may discretize the spatial domain on the empirical node set or on a regular grid, define a quadratic operator controlling field covariance, and tune the coefficients so that the resulting covariance function matches the empirical graph kernel. The quartic term may be treated approximately, for example through mean-field, perturbative, or numerical fitting assumptions, provided that the approximation is explicitly documented.

Optional extensions (up to +0.5):

- apply the same pipeline to both European and North-American GridKit networks;
- compare power grids with another spatial network class;
- test whether fitted coefficients differ between dense and sparse regions;
- compare homogeneous, non-homogeneous, and purely topological kernels;
- generate synthetic networks from the inferred kernels and compare their degree distribution, clustering, and component structure with the empirical network;
- explore whether the fitted field-theoretic parameters can be used as compact descriptors of spatial network organization.

This is primarily a methodological project. The goal is not to produce a complete field theory of spatial networks, but to implement and test a minimal computational bridge between empirical graph kernels and covariance functions from statistical field theory.

## About the code provided at the beginning of the project

Note that we provide example code for the separate tasks of kernel fitting, and sampling fields from the Ginzburg-Landau theory.
This is intended as bootstrap material, and require considerable refinement: the student is expected to produce optimized code, possibly exploring alternative algorithms.