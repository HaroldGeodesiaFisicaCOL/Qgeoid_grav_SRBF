# Introduction Qgeoid_grav_SRBF
`Qgeoid_grav_SRBF` is a Python package for regional gravity field refinement using Spherical Radial Basis Functions (SRBFs). Features VCE-based regularization of the system of equations and optional Tikhonov parameter optimization via the L-curve criterion for stable and accurate quasigeoid/gravity field modeling.

## Features for modules
### Download of the geopotential models 
- Dowloand the spherical harmonics coefficients from the ICGEM website of the geopotential model in .gfc format.
- Dowloand the spherical harmonics coefficients from the ICGEM website of the topographic gravitational effects model (dv_ell_Earth2014) in .gfc format.
- Dowloand the grid model of the topografic gravitational effects of the model ERTM2190 in .tif format.

### SRBF gravity field refinement
- $Nmax_i$ Kernel of SRBFs expansion.
- Uses the Shannon band pass filter for the terrestrial data and Cup (Cubic polynomial) for the Aerogravimetric data.
- Construction of the $y = Ad$ system of equations.
- Analysis step (Estimation of parameters) using SRBFs with VCE-based regularization.
- Analysis step (Estimation of parameters) using SRBFs with VCE-based regularization and Tikhonov parameter optimization.
- Sintesys step (Estimation of residual perturbance potential) using SRBF's and system $y = B \hat{d}$.
- Restore procedure to determine the Qgeoid model and the standart deviation.

### Results visualization
- Visualization of the estimated parameters and the standard deviation.
- Visualization of the significant parameters.
- Visualization of the estimated residual perturbance potential and the standard deviation.
- Visualization of the estimated Qgeoid model and the standard deviation.

# Documentation
The documentation can be found in: 
- Theorical aspects, [docs/Qgeoid_grav_SRBF.pdf](docs/Qgeoid_grav_SRBF.pdf)
- Example of aplication (Results), [docs\Technical_Report_QgeoidMEDE2026.pdf](docs\Technical_Report_QgeoidMEDE2026.pdf)

# Contact
Feel free to contact the authors: 
* Harold Olarte Ramírez, at geodesiafisica2208@gmail.com.
* Carlos Rico Acevedo, at caarlosrico@gmail.com.

# Citing
* Olarte, H. S. (2025). Estimación de un modelo micro-cuasigeoidal mediante el modelamiento del campo de gravedad local para la incorporación de la primera estación de Colombia en el IHRF. Recuperado de: http://hdl.handle.net/11349/100030
