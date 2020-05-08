Bayesian Zero- and- One Inflated Beta Regression in Stan with Causal Mediation
================

Purpose
-------

This code implements a non-parametric Bayesian appraoch to causal mediation anaylsis, using a zero-one inflated beta regression model in STAN.To illustrate the method, the JOBS II data is used. This dataset can be found in the R package [`mediation`](https://cran.r-project.org/web/packages/mediation/index.html).The JOBS II study was a randomized field experiment that investigated the efficacy of a job training intervention on unemployed workers. In this dataset there are 899 observations, containing no missing values. The potential outcome variable, `depress2`, is a continuous measure of depressive symptoms post-treatment (participation in job-skills workshops). The mediator variable, `job_seek`, is a continuous measure of job search self-efficacy. The treatment variable, `treat` is an indicator variable for whether the participant was randomly selected for the JOBS II training program.

Pre-process data
----------------

For pre-processing of the data, the predictors are scaled using the R function `scale`. The scaled matrices are designed to include a column for the intercept. Both the mediator and the outcome variables are scaled to be between 0 and 1.

``` r
library(rstan)
library(mediation)
library(ggplot2)

data(jobs); invisible(names(jobs))

normalize <- function(x){ 
  return((x- min(x)) /(max(x)-min(x)))
}

## create data 
scaled_z <- scale(jobs[,c('econ_hard','sex','age')])
trt <- jobs$treat

## scale mediate and outcome
y = normalize(jobs$depress2)
med = normalize(jobs$job_seek)

qplot(y, geom = 'histogram', binwidth = .025, ylab = 'Frequency', xlab = 'Measure of Depression', fill=I('white'), col=I('blue')) + theme_bw() + theme(panel.grid.minor = element_blank())
```

![](README_files/figure-markdown_github/preprocess_rdata,%20ggplot2-1.png)

``` r
qplot(med, geom = 'histogram', binwidth = .025, ylab = 'Frequency', xlab = 'Measure of Confidence/Self- Efficacy in Job Search', fill=I('white'), col=I('blue')) + theme_bw() + theme(panel.grid.minor = element_blank())
```

![](README_files/figure-markdown_github/preprocess_rdata,%20ggplot2-2.png)

Statistical Model
-----------------

Both the response variables, `depress2` and `job_seek` appear to follow a beta distribution and lie within the bounds \[0,1\]. Since 0 and 1 are included in the interval, we will model the data using a zero- and- one inflated beta regression model.

$$f(response;\\alpha\_\*, \\gamma\_\*, \\mu\_\*, \\phi\_\*)
  \\begin{cases}
        \\alpha\_\*(1-\\gamma\_\*) & \\text{if  y = 0} \\\\
        \\alpha\_\*\\gamma\_\* & \\text{if  y = 1} \\\\
        (1-\\alpha\_\*)f(response;\\mu\_\*,\\phi\_\*) & \\text{if  y } \\in \\text{(0,1)} \\\\
  \\end{cases}$$

Where $ 0 &lt;, , 0, ;; f(response;**,**)~ Beta(p\_*= **, q\_*= \_\*(1-\_\*)) $

Store data and input in a list to send to STAN
----------------------------------------------

The STAN model accepts the following values stored in a list:

    * n - the total number of observations

    * np - the total number of predictors,excluding the intercept

    * sim - the total number of iterations per chain
        
    * y - the outcome variable scaled between 0 and 1; vector

    * m - the mediator variable scaled between 0 and 1; vector

    * a - the treatment variable; vector

    * z - the data matrix of scaled predictors

    * alpha_cov_m - the covariance for the normal prior set on alpha; used to model m

    * gamma_cov_m -  the covariance for the normal prior set on gamma; used to model m

    * mu_cov_m -  the covariance for the normal prior set on mu; used to model m

    * phi_cov_m -  the covariance for the normal prior set on phi; used to model m

    * alpha_cov_y - the covariance for the normal prior set on alpha; used to model y

    * gamma_cov_y -  the covariance for the normal prior set on gamma; used to model y

    * mu_cov_y -  the covariance for the normal prior set on mu; used to model y

    * phi_cov_y -  the covariance for the normal prior set on phi; used to model y

``` r
jobs_data <-
  list(n = nrow(scaled_z),
       np = ncol(scaled_z), ## number of parameters excluding intercept
       sim = 1000,
       y = y,
       m = med,
       a = trt,
       z = scaled_z,    
       ## cov_m: prior for coefficients of the mediatior model; include treatment, do NOT include the intercept or mediator
       alpha_cov_m = 5*diag(1, ncol(scaled_z)+1), ## == np + 1
       gamma_cov_m = 5*diag(1, ncol(scaled_z)+1),
       mu_cov_m = 5*diag(1, ncol(scaled_z)+1),
       phi_cov_m = 5*diag(1, ncol(scaled_z)+1),
       ## cov_y: prior for coefficients of the outcome model; include the mediator and treatment, do not include the intercept
       alpha_cov_y = 5*diag(1, ncol(scaled_z)+2),  ## == np + 2
       gamma_cov_y = 5*diag(1, ncol(scaled_z)+2),
       mu_cov_y = 5*diag(1, ncol(scaled_z)+2),
       phi_cov_y = 5*diag(1, ncol(scaled_z)+2)
  )
```

Stan Model
----------

This model will return:

    * all_params_y - alpha, gamma, p, q for the outcome model (1:nchains*sim,1:n,1:4)

    * all_params_m - alpha, gamma, p, q for the mediator model (1:nchains*sim,1:n,1:4)

    * coef_mediator -  alpha, gamma, mu, phi;  coefficients for the mediator model (1:nchains*sim,1:np,1:4)

    * coef_outcome -  alpha, gamma, mu, phi; coefficients for the outcome model (1:nchains*sim,1:np+1,1:4)

    * tau - total effect (length = nchains * sim)

    * delta - causal effect (1:nchains*sim, 2) where [t= 0, t = 1]

    * zeta - direct effect (1:nchains*sim, 2) where [t= 0, t = 1]

We can fit the model in Stan with the following code:
