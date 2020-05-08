Bayesian Zero- and- One Inflated Beta Regression in Stan with Causal
Mediation
================
Lexi Rene

## Purpose

This code implements a Bayesian appraoch to causal mediation anaylsis,
using a zero-one inflated beta regression model in STAN. To illustrate
the method, the JOBS II data is used. This dataset can be found in the R
package
[`mediation`](https://cran.r-project.org/web/packages/mediation/index.html).
The JOBS II study was a randomized field experiment that investigated
the efficacy of a job training intervention on unemployed workers. In
this dataset there are 899 observations, containing no missing values.
The potential outcome variable, `depress2`, is a continuous measure of
depressive symptoms post-treatment (participation in job-skills
workshops). The mediator variable, `job_seek`, is a continuous measure
of job search self-efficacy. The treatment variable, `treat` is an
indicator variable for whether the participant was randomly selected for
the JOBS II training program.

## Pre-process data

For pre-processing of the data, the predictors are scaled using the R
function `scale`. The scaled matrices are designed to include a column
for the intercept. Both the mediator and the outcome variables are
scaled to be between 0 and 1.

``` r
library(rstan)
library(mediation)
library(ggplot2)
library(tictoc)
library(bayesplot)
library(kableExtra)
library(ggpubr)
# library(shinystan)

data(jobs); invisible(names(jobs))

normalize <- function(x){ 
  return((x- min(x)) /(max(x)-min(x)))
}
round_df <- function(x, digits) {
    # round all numeric variables
    # x: data frame 
    # digits: number of digits to round
    numeric_columns <- sapply(x, class) == 'numeric'
    x[numeric_columns] <-  round(x[numeric_columns], digits)
    x
}
num_seed = 1810201

## create data 
scaled_z <- scale(jobs[,c('econ_hard','sex','age')])
trt <- jobs$treat

## scale mediate and outcome
y <- normalize(jobs$depress2)
med <- normalize(jobs$job_seek)

qplot(y, geom = 'histogram', binwidth = .025, ylab = 'Frequency', xlab = 'Measure of Depression', fill=I('white'), col=I('blue')) + theme_bw() + theme(panel.grid.minor = element_blank())
```

![](README_files/figure-gfm/preprocess_rdata,%20ggplot2-1.png)<!-- -->

``` r
qplot(med, geom = 'histogram', binwidth = .025, ylab = 'Frequency', xlab = 'Measure of Confidence/Self- Efficacy in Job Search', fill=I('white'), col=I('blue')) + theme_bw() + theme(panel.grid.minor = element_blank())
```

![](README_files/figure-gfm/preprocess_rdata,%20ggplot2-2.png)<!-- -->

## Causal Mediation Analysis

### Potential/ Counterfactual Framework

Using the potential (counterfactual) outcome framework, the causal
effect of the job training program can be defined as the difference
between two potential outcomes. One potential outcome is realized if the
subject participates in the training program and the other potential
outcome is realized if the subject does not participate.

Suppose we use ![Y\_i](https://ibm.codecogs.com/png.latex?Y_i "Y_i") to
represent the measure of depression for the
![i^{th}](https://ibm.codecogs.com/png.latex?i%5E%7Bth%7D "i^{th}")
subject, ![M\_i](https://ibm.codecogs.com/png.latex?M_i "M_i"), the
measure of confidence/self efficiacy in the job search for the
![i^{th}](https://ibm.codecogs.com/png.latex?i%5E%7Bth%7D "i^{th}")
subject, and ![A\_i](https://ibm.codecogs.com/png.latex?A_i "A_i"), the
binary indicator variable for the
![i^{th}](https://ibm.codecogs.com/png.latex?i%5E%7Bth%7D "i^{th}")
subject’s treatment/participation in the JOBS II training program;
![A\_i](https://ibm.codecogs.com/png.latex?A_i "A_i") takes on the
values of ![a= 1](https://ibm.codecogs.com/png.latex?a%3D%201 "a= 1")
(participation in the training) or ![a
= 0](https://ibm.codecogs.com/png.latex?a%20%3D%200 "a = 0")
(otherwise). The depression level of subject
![i](https://ibm.codecogs.com/png.latex?i "i") is independent of subject
![j](https://ibm.codecogs.com/png.latex?j "j") (![i \\ne
j](https://ibm.codecogs.com/png.latex?i%20%5Cne%20j "i \\ne j")). In
addition, since the treatment in the JOBS II study is randomized,
![A\_i](https://ibm.codecogs.com/png.latex?A_i "A_i") is statistically
independent of the potential outcomes; we can write this as
![\\left(Y\_i^{A\_i = 0},Y\_i^{A\_i = 1}\\right) \\perp \\\!\\\!\\\!
\\perp{}
A\_i](https://ibm.codecogs.com/png.latex?%5Cleft%28Y_i%5E%7BA_i%20%3D%200%7D%2CY_i%5E%7BA_i%20%3D%201%7D%5Cright%29%20%5Cperp%20%5C%21%5C%21%5C%21%20%5Cperp%7B%7D%20A_i
"\\left(Y_i^{A_i = 0},Y_i^{A_i = 1}\\right) \\perp \\!\\!\\! \\perp{} A_i").
The observed value for the depression level can be denoted by
![Y\_i](https://ibm.codecogs.com/png.latex?Y_i "Y_i"), where ![Y\_i =
Y\_i^{A\_i}](https://ibm.codecogs.com/png.latex?Y_i%20%3D%20Y_i%5E%7BA_i%7D
"Y_i = Y_i^{A_i}"), which can result in two potential values. For
example,
![Y\_i^{A\_i=1}](https://ibm.codecogs.com/png.latex?Y_i%5E%7BA_i%3D1%7D
"Y_i^{A_i=1}") would be the observed depression level for subject
![i](https://ibm.codecogs.com/png.latex?i "i"), if subject
![i](https://ibm.codecogs.com/png.latex?i "i") actually participated in
the training program; in this case, the unobserved outcome for subject
![i](https://ibm.codecogs.com/png.latex?i "i") is the level of
depression if they did not participate in the training program. We will
use ![Y\_i^{A\_i =
a}](https://ibm.codecogs.com/png.latex?Y_i%5E%7BA_i%20%3D%20a%7D
"Y_i^{A_i = a}") to represent the potential level of depression that
would result under the treatment status
![a](https://ibm.codecogs.com/png.latex?a "a") for subject
![i](https://ibm.codecogs.com/png.latex?i "i"). In addition, in causal
mediation analysis, the potential outcome also depends on the mediator.
In the context of this study, this implies that the level of job search
self-efficacy can be affected by participation in the program, which can
be represented by
![M\_i^{A\_i=a}](https://ibm.codecogs.com/png.latex?M_i%5E%7BA_i%3Da%7D
"M_i^{A_i=a}"); which also has two potential values
![M\_i^{A\_i=0}](https://ibm.codecogs.com/png.latex?M_i%5E%7BA_i%3D0%7D
"M_i^{A_i=0}") and
![M\_i^{A\_i=1}](https://ibm.codecogs.com/png.latex?M_i%5E%7BA_i%3D1%7D
"M_i^{A_i=1}"). The potential mediator value of subject
![i](https://ibm.codecogs.com/png.latex?i "i") are independent of the
treatment status for subject ![j](https://ibm.codecogs.com/png.latex?j
"j") (![i \\ne j](https://ibm.codecogs.com/png.latex?i%20%5Cne%20j
"i \\ne j")). Therefore, we will update the potential outcome to be
denoted as ![Y\_i^{A\_i, M\_i^{A\_i
}}](https://ibm.codecogs.com/png.latex?Y_i%5E%7BA_i%2C%20M_i%5E%7BA_i%20%7D%7D
"Y_i^{A_i, M_i^{A_i }}") and also note that the potential outcome for
subject ![i](https://ibm.codecogs.com/png.latex?i "i") is indepedent of
both the treatment status and the mediator value of subject
![j](https://ibm.codecogs.com/png.latex?j "j") (![i \\ne
j](https://ibm.codecogs.com/png.latex?i%20%5Cne%20j "i \\ne j")).

### Causal Mediation Effects

The statistical independence between the treatment and the potential
outcome allows us to compute the average causal effect as the observed
mean difference between the treatment and control group:

<center>

  
![E\[Y\_i^{A\_i=1}-Y\_i^{A\_i=0}\] = E\[Y\_i^{A\_i=1}|A\_i = 1\] -
E\[Y\_i^{A\_i=0}|A\_i = 0\] \\quad (average \\;causal \\;
effect)](https://ibm.codecogs.com/png.latex?E%5BY_i%5E%7BA_i%3D1%7D-Y_i%5E%7BA_i%3D0%7D%5D%20%3D%20E%5BY_i%5E%7BA_i%3D1%7D%7CA_i%20%3D%201%5D%20-%20E%5BY_i%5E%7BA_i%3D0%7D%7CA_i%20%3D%200%5D%20%5Cquad%20%28average%20%5C%3Bcausal%20%5C%3B%20effect%29
"E[Y_i^{A_i=1}-Y_i^{A_i=0}] = E[Y_i^{A_i=1}|A_i = 1] - E[Y_i^{A_i=0}|A_i = 0] \\quad (average \\;causal \\; effect)")  

</center>

Under the counterfactual/potential outcome framework, only one potential
outcome of ![Y\_i^{A\_i=a, M\_i^{A\_i=a
}](https://ibm.codecogs.com/png.latex?Y_i%5E%7BA_i%3Da%2C%20M_i%5E%7BA_i%3Da%20%7D
"Y_i^{A_i=a, M_i^{A_i=a }") is observed. Let
![Z\_i](https://ibm.codecogs.com/png.latex?Z_i "Z_i") be a vector of
baseline covariate for each subject
![i](https://ibm.codecogs.com/png.latex?i "i") and ![\\mathscr
Z](https://ibm.codecogs.com/png.latex?%5Cmathscr%20Z "\\mathscr Z") be
the support of the distribution of
![Z\_i](https://ibm.codecogs.com/png.latex?Z_i "Z_i"); in addition, the
support of ![M\_i](https://ibm.codecogs.com/png.latex?M_i "M_i") is
![\\mathscr{M}](https://ibm.codecogs.com/png.latex?%5Cmathscr%7BM%7D
"\\mathscr{M}"). To identify the effects of treatment and mediation, we
assume sequential ignorability, as per Imani et al, by assuming the
following two statements of conditional independence hold: <br />

<center>

  
![\\Bigl \\{ Y\_i^{A\_i = a', m = M\_i^{A\_i = a'}}, \\; M\_i^{A\_i=a}
\\Bigr \\} \\perp \\\!\\\!\\\! \\perp{} A\_i|Z\_i=z \\qquad (1) \\\\
Y\_i^{A\_i=a', m = M\_i^{A\_i=a'}} \\perp \\\!\\\!\\\! \\perp{}
M\_i|A\_i,Z\_i = z \\qquad
(2)](https://ibm.codecogs.com/png.latex?%5CBigl%20%5C%7B%20Y_i%5E%7BA_i%20%3D%20a%27%2C%20m%20%3D%20M_i%5E%7BA_i%20%3D%20a%27%7D%7D%2C%20%5C%3B%20M_i%5E%7BA_i%3Da%7D%20%5CBigr%20%5C%7D%20%5Cperp%20%5C%21%5C%21%5C%21%20%5Cperp%7B%7D%20A_i%7CZ_i%3Dz%20%5Cqquad%20%20%281%29%20%5C%5C%0AY_i%5E%7BA_i%3Da%27%2C%20m%20%3D%20M_i%5E%7BA_i%3Da%27%7D%7D%20%5Cperp%20%5C%21%5C%21%5C%21%20%5Cperp%7B%7D%20M_i%7CA_i%2CZ_i%20%3D%20z%20%5Cqquad%20%282%29
"\\Bigl \\{ Y_i^{A_i = a', m = M_i^{A_i = a'}}, \\; M_i^{A_i=a} \\Bigr \\} \\perp \\!\\!\\! \\perp{} A_i|Z_i=z \\qquad  (1) \\\\
Y_i^{A_i=a', m = M_i^{A_i=a'}} \\perp \\!\\!\\! \\perp{} M_i|A_i,Z_i = z \\qquad (2)")  

</center>

where ![P(A\_i=a|Z\_i=z)
\> 0](https://ibm.codecogs.com/png.latex?P%28A_i%3Da%7CZ_i%3Dz%29%20%3E%200
"P(A_i=a|Z_i=z) \> 0") and ![P(m=M\_i^{A\_i = a}|A\_i=a,Z\_i=z)
\> 0](https://ibm.codecogs.com/png.latex?P%28m%3DM_i%5E%7BA_i%20%3D%20a%7D%7CA_i%3Da%2CZ_i%3Dz%29%20%3E%200
"P(m=M_i^{A_i = a}|A_i=a,Z_i=z) \> 0") for ![a
= 0, 1](https://ibm.codecogs.com/png.latex?a%20%3D%200%2C%201
"a = 0, 1"), and, all ![z \\in
\\mathscr{Z}](https://ibm.codecogs.com/png.latex?z%20%5Cin%20%5Cmathscr%7BZ%7D
"z \\in \\mathscr{Z}") and ![m \\in
\\mathscr{M}](https://ibm.codecogs.com/png.latex?m%20%5Cin%20%5Cmathscr%7BM%7D
"m \\in \\mathscr{M}").

These ignorability assumptions are made sequentially. The first part of
the assumption assumes that given the observed confounders, prior to
treatment, the treatment assignment is ignorable. In terms of
statistical independence, the observed pre-treatment is independent of
the potential outcomes and potential mediators. The second part of
sequential ignorability states that the mediator is ignorable given the
observed treatment and pre-treatment confounders; meaning that the
potential outcome and mediator are unconfounded on the past observations
and confounders.

The indirect effect of the treatment on the outcome, through the
mediating variable is defined as the causal mediation effect (Imai et
al., 2010), for ![a
= 0, 1](https://ibm.codecogs.com/png.latex?a%20%3D%200%2C%201
"a = 0, 1"):

<center>

  
![\\quad \\delta\_i^{A\_i=a} \\equiv Y\_i^{A\_i=a, M\_i^{A\_i=1 }} -
Y\_i^{A\_i=a, M\_i^{A\_i=0 }} \\\\ (causal \\; mediation/ natural\\;
indirect \\;
effect)](https://ibm.codecogs.com/png.latex?%5Cquad%20%5Cdelta_i%5E%7BA_i%3Da%7D%20%5Cequiv%20Y_i%5E%7BA_i%3Da%2C%20M_i%5E%7BA_i%3D1%20%7D%7D%20-%20Y_i%5E%7BA_i%3Da%2C%20M_i%5E%7BA_i%3D0%20%7D%7D%20%20%5C%5C%20%28causal%20%5C%3B%20mediation%2F%20natural%5C%3B%20indirect%20%5C%3B%20effect%29
"\\quad \\delta_i^{A_i=a} \\equiv Y_i^{A_i=a, M_i^{A_i=1 }} - Y_i^{A_i=a, M_i^{A_i=0 }}  \\\\ (causal \\; mediation/ natural\\; indirect \\; effect)")  

</center>

The following definitions are defined as the effect of the treatment
![(a
= 0, 1)](https://ibm.codecogs.com/png.latex?%28a%20%3D%200%2C%201%29
"(a = 0, 1)"), on the outcome, through the mediation variable:

The average causal mediation effect is defined by:

<center>

  
![\\qquad \\overline{\\delta^{A=a}} \\equiv E\\left\[Y\_i^{A\_i=a,
M\_i^{A\_i=1 }} - Y\_i^{A\_i=a, M\_i^{A\_i=0 }}\\right\] \\\\ 
(average \\; causal \\; mediation/ natural\\; indirect \\;
effect)](https://ibm.codecogs.com/png.latex?%5Cqquad%20%5Coverline%7B%5Cdelta%5E%7BA%3Da%7D%7D%20%5Cequiv%20E%5Cleft%5BY_i%5E%7BA_i%3Da%2C%20M_i%5E%7BA_i%3D1%20%7D%7D%20-%20Y_i%5E%7BA_i%3Da%2C%20M_i%5E%7BA_i%3D0%20%7D%7D%5Cright%5D%20%20%5C%5C%20%0A%28average%20%5C%3B%20causal%20%5C%3B%20mediation%2F%20natural%5C%3B%20indirect%20%5C%3B%20effect%29
"\\qquad \\overline{\\delta^{A=a}} \\equiv E\\left[Y_i^{A_i=a, M_i^{A_i=1 }} - Y_i^{A_i=a, M_i^{A_i=0 }}\\right]  \\\\ 
(average \\; causal \\; mediation/ natural\\; indirect \\; effect)")  

</center>

The direct effect is defined by:

<center>

  
![\\zeta\_i^{A\_i=a} \\equiv Y\_i^{A\_i=1, M\_i^{A\_i=a }} -
Y\_i^{A\_i=0, M\_i^{A\_i=a }} \\\\
\\qquad (natural\\; direct \\;
effect)](https://ibm.codecogs.com/png.latex?%5Czeta_i%5E%7BA_i%3Da%7D%20%5Cequiv%20Y_i%5E%7BA_i%3D1%2C%20M_i%5E%7BA_i%3Da%20%7D%7D%20-%20Y_i%5E%7BA_i%3D0%2C%20M_i%5E%7BA_i%3Da%20%7D%7D%20%20%5C%5C%0A%5Cqquad%20%28natural%5C%3B%20direct%20%5C%3B%20effect%29
"\\zeta_i^{A_i=a} \\equiv Y_i^{A_i=1, M_i^{A_i=a }} - Y_i^{A_i=0, M_i^{A_i=a }}  \\\\
\\qquad (natural\\; direct \\; effect)")  

</center>

The average direct effect is defined by:

<center>

  
![\\overline{\\zeta^{A=a}} \\equiv E\\left\[Y\_i^{A\_i=1, M\_i^{A\_i=a
}} - Y\_i^{A\_i=0, M\_i^{A\_i=a }}\\right\] \\\\ 
\\qquad (average\\; natural\\; direct \\;
effect)](https://ibm.codecogs.com/png.latex?%5Coverline%7B%5Czeta%5E%7BA%3Da%7D%7D%20%5Cequiv%20E%5Cleft%5BY_i%5E%7BA_i%3D1%2C%20M_i%5E%7BA_i%3Da%20%7D%7D%20-%20Y_i%5E%7BA_i%3D0%2C%20M_i%5E%7BA_i%3Da%20%7D%7D%5Cright%5D%20%20%5C%5C%20%0A%5Cqquad%20%28average%5C%3B%20natural%5C%3B%20direct%20%5C%3B%20effect%29
"\\overline{\\zeta^{A=a}} \\equiv E\\left[Y_i^{A_i=1, M_i^{A_i=a }} - Y_i^{A_i=0, M_i^{A_i=a }}\\right]  \\\\ 
\\qquad (average\\; natural\\; direct \\; effect)")  

</center>

The total effect is defined by:

<center>

  
![\\tau\_i \\equiv Y\_i^{A\_i=1, M\_i^{A\_i=1 }} - Y\_i^{A\_i=0,
M\_i^{A\_i=0 }} \\\\
\\qquad (total \\;
effect)](https://ibm.codecogs.com/png.latex?%5Ctau_i%20%5Cequiv%20Y_i%5E%7BA_i%3D1%2C%20M_i%5E%7BA_i%3D1%20%7D%7D%20-%20Y_i%5E%7BA_i%3D0%2C%20M_i%5E%7BA_i%3D0%20%7D%7D%20%20%5C%5C%0A%5Cqquad%20%28total%20%5C%3B%20effect%29
"\\tau_i \\equiv Y_i^{A_i=1, M_i^{A_i=1 }} - Y_i^{A_i=0, M_i^{A_i=0 }}  \\\\
\\qquad (total \\; effect)")  

</center>

Lastly, the average total effect is defined by:

<center>

  
![\\overline{\\tau} \\equiv E\\left\[Y\_i^{A\_i=1, M\_i^{A\_i=1 }} -
Y\_i^{A\_i=0, M\_i^{A\_i=0 }}\\right\] \\\\ 
\\qquad (average \\; total \\;
effect)](https://ibm.codecogs.com/png.latex?%5Coverline%7B%5Ctau%7D%20%5Cequiv%20E%5Cleft%5BY_i%5E%7BA_i%3D1%2C%20M_i%5E%7BA_i%3D1%20%7D%7D%20-%20Y_i%5E%7BA_i%3D0%2C%20M_i%5E%7BA_i%3D0%20%7D%7D%5Cright%5D%20%20%5C%5C%20%0A%5Cqquad%20%28average%20%5C%3B%20total%20%5C%3B%20effect%29
"\\overline{\\tau} \\equiv E\\left[Y_i^{A_i=1, M_i^{A_i=1 }} - Y_i^{A_i=0, M_i^{A_i=0 }}\\right]  \\\\ 
\\qquad (average \\; total \\; effect)")  

</center>

Under the assumptions from
![(1)](https://ibm.codecogs.com/png.latex?%281%29 "(1)") and
![(2)](https://ibm.codecogs.com/png.latex?%282%29 "(2)"), Imai et
al. then showed that the distribution of the potential outcomes is
nonparametrically identified:

<center>

  
![\\begin{align\*} f \\left( Y\_i^{A\_i, M\_i^{A\_i= a'}} |Z\_i = z
\\right) = \\int\_M f(Y\_i|M\_i = m, A\_i = a, Z\_i =
z)dF\_{M\_i}(m|A\_i=a',Z\_i=z) \\quad x \\in X; \\; a,a' = 0,1 \\qquad
(3)
\\end{align\*}](https://ibm.codecogs.com/png.latex?%5Cbegin%7Balign%2A%7D%20f%20%5Cleft%28%20Y_i%5E%7BA_i%2C%20M_i%5E%7BA_i%3D%20a%27%7D%7D%20%7CZ_i%20%3D%20z%20%5Cright%29%20%3D%20%5Cint_M%20f%28Y_i%7CM_i%20%3D%20m%2C%20A_i%20%3D%20a%2C%20Z_i%20%3D%20z%29dF_%7BM_i%7D%28m%7CA_i%3Da%27%2CZ_i%3Dz%29%20%5Cquad%20x%20%5Cin%20X%3B%20%5C%3B%20a%2Ca%27%20%3D%200%2C1%20%5Cqquad%20%283%29%20%5Cend%7Balign%2A%7D
"\\begin{align*} f \\left( Y_i^{A_i, M_i^{A_i= a'}} |Z_i = z \\right) = \\int_M f(Y_i|M_i = m, A_i = a, Z_i = z)dF_{M_i}(m|A_i=a',Z_i=z) \\quad x \\in X; \\; a,a' = 0,1 \\qquad (3) \\end{align*}")  

</center>

This result allows us to estimate the potential outcome and mediators
that we are unable to observe.

## Statistical Model

#### Densities

The density of a random variable
![R](https://ibm.codecogs.com/png.latex?R "R") with a beta distribution,
where ![0 \< r
\< 1](https://ibm.codecogs.com/png.latex?0%20%3C%20r%20%3C%201
"0 \< r \< 1"), can be reparametrized (Ferrari & Cribari-Neto 2004) to
be defined as:

<center>

  
![f\_{beta}(r;\\mu,\\phi) =
\\frac{\\Gamma(\\phi)}{\\Gamma(\\mu\\phi)\\Gamma((1-\\mu)\\phi)}r^{\\mu\\phi}(1-r)^{(1-\\mu)\\phi-1}
\\quad(6)](https://ibm.codecogs.com/png.latex?f_%7Bbeta%7D%28r%3B%5Cmu%2C%5Cphi%29%20%3D%20%5Cfrac%7B%5CGamma%28%5Cphi%29%7D%7B%5CGamma%28%5Cmu%5Cphi%29%5CGamma%28%281-%5Cmu%29%5Cphi%29%7Dr%5E%7B%5Cmu%5Cphi%7D%281-r%29%5E%7B%281-%5Cmu%29%5Cphi-1%7D%20%5Cquad%286%29
"f_{beta}(r;\\mu,\\phi) = \\frac{\\Gamma(\\phi)}{\\Gamma(\\mu\\phi)\\Gamma((1-\\mu)\\phi)}r^{\\mu\\phi}(1-r)^{(1-\\mu)\\phi-1} \\quad(6)")  

</center>

<em> Note for ![(6)](https://ibm.codecogs.com/png.latex?%286%29 "(6)"):
![\\Gamma(\\cdot)](https://ibm.codecogs.com/png.latex?%5CGamma%28%5Ccdot%29
"\\Gamma(\\cdot)") denotes the gamma function, ![E\[R\] =
\\mu](https://ibm.codecogs.com/png.latex?E%5BR%5D%20%3D%20%5Cmu
"E[R] = \\mu"), and ![Var(R) =
\\frac{\\mu(1-\\mu)}{\\phi+1}](https://ibm.codecogs.com/png.latex?Var%28R%29%20%3D%20%5Cfrac%7B%5Cmu%281-%5Cmu%29%7D%7B%5Cphi%2B1%7D
"Var(R) = \\frac{\\mu(1-\\mu)}{\\phi+1}");
![\\phi](https://ibm.codecogs.com/png.latex?%5Cphi "\\phi") is a
precision parameter, where for a fixed
![\\mu](https://ibm.codecogs.com/png.latex?%5Cmu "\\mu"), there is an
inverse relationship between
![\\phi](https://ibm.codecogs.com/png.latex?%5Cphi "\\phi") and
![Var(R)](https://ibm.codecogs.com/png.latex?Var%28R%29 "Var(R)").</em>

Using ![(3)](https://ibm.codecogs.com/png.latex?%283%29 "(3)") we
further assume the following regression models for both the response
variables, `depress2`, ![y](https://ibm.codecogs.com/png.latex?y "y"),
and `job_seek`, ![m](https://ibm.codecogs.com/png.latex?m "m"), to
follow a zero-one inflated beta (ZOIB) distribution, as they lie within
the bounds \[0,1\]. The cumulative distribution function of the random
variable ![R](https://ibm.codecogs.com/png.latex?R "R") under a ZOIB
distribution is:

<center>

![\\begin{align\*} F\_{ZOIB}(R;\\alpha, \\gamma, \\mu, \\phi) = \\alpha
\\; \\text{I}(r=0) + (1-\\alpha)\\gamma \\; \\text{I}(r = 1) +
(1-\\alpha)(1-\\gamma)F\_{beta}(R;\\mu,\\phi)
\\end{align\*}](https://ibm.codecogs.com/png.latex?%5Cbegin%7Balign%2A%7D%20F_%7BZOIB%7D%28R%3B%5Calpha%2C%20%5Cgamma%2C%20%5Cmu%2C%20%5Cphi%29%20%3D%20%5Calpha%20%5C%3B%20%5Ctext%7BI%7D%28r%3D0%29%20%2B%20%281-%5Calpha%29%5Cgamma%20%5C%3B%20%5Ctext%7BI%7D%28r%20%3D%201%29%20%2B%20%281-%5Calpha%29%281-%5Cgamma%29F_%7Bbeta%7D%28R%3B%5Cmu%2C%5Cphi%29%20%5Cend%7Balign%2A%7D
"\\begin{align*} F_{ZOIB}(R;\\alpha, \\gamma, \\mu, \\phi) = \\alpha \\; \\text{I}(r=0) + (1-\\alpha)\\gamma \\; \\text{I}(r = 1) + (1-\\alpha)(1-\\gamma)F_{beta}(R;\\mu,\\phi) \\end{align*}")

</center>

Where the density of ![R](https://ibm.codecogs.com/png.latex?R "R"),is
defined as:

<center>

  
![f\_{ZOIB}(r\_i;\\alpha, \\gamma, \\mu, \\phi) =
\\begin{cases}
\\alpha & \\text{if } r\_i = 0 \\\\
(1-\\alpha)\\gamma & \\text{if } r\_i = 1 \\\\
(1-\\alpha)(1-\\gamma)f\_{beta}(r\_i;\\mu\_{i},\\phi\_{i}) & \\text{if
}r\_i \\in \\text{(0,1)} \\\\
\\end{cases}
\\quad
(7)](https://ibm.codecogs.com/png.latex?f_%7BZOIB%7D%28r_i%3B%5Calpha%2C%20%5Cgamma%2C%20%5Cmu%2C%20%5Cphi%29%20%3D%0A%20%20%5Cbegin%7Bcases%7D%0A%20%20%20%20%20%20%20%20%5Calpha%20%26%20%5Ctext%7Bif%20%7D%20r_i%20%3D%200%20%5C%5C%0A%20%20%20%20%20%20%20%20%281-%5Calpha%29%5Cgamma%20%26%20%5Ctext%7Bif%20%20%7D%20r_i%20%3D%201%20%5C%5C%0A%20%20%20%20%20%20%20%20%281-%5Calpha%29%281-%5Cgamma%29f_%7Bbeta%7D%28r_i%3B%5Cmu_%7Bi%7D%2C%5Cphi_%7Bi%7D%29%20%26%20%5Ctext%7Bif%20%20%7Dr_i%20%5Cin%20%5Ctext%7B%280%2C1%29%7D%20%5C%5C%0A%20%20%5Cend%7Bcases%7D%0A%20%20%5Cquad%20%287%29
"f_{ZOIB}(r_i;\\alpha, \\gamma, \\mu, \\phi) =
  \\begin{cases}
        \\alpha & \\text{if } r_i = 0 \\\\
        (1-\\alpha)\\gamma & \\text{if  } r_i = 1 \\\\
        (1-\\alpha)(1-\\gamma)f_{beta}(r_i;\\mu_{i},\\phi_{i}) & \\text{if  }r_i \\in \\text{(0,1)} \\\\
  \\end{cases}
  \\quad (7)")  

![{0 \< \\alpha, \\gamma, \\mu \< 1 ,\\;\\; \\phi \> 0, \\;\\;
f\_{beta}(r;\\mu,\\phi) \\sim Beta(p= \\mu\\phi,\\: q=
\\phi(1-\\mu))}](https://ibm.codecogs.com/png.latex?%7B0%20%3C%20%5Calpha%2C%20%5Cgamma%2C%20%5Cmu%20%3C%201%20%2C%5C%3B%5C%3B%20%5Cphi%20%3E%200%2C%20%5C%3B%5C%3B%20f_%7Bbeta%7D%28r%3B%5Cmu%2C%5Cphi%29%20%5Csim%20Beta%28p%3D%20%5Cmu%5Cphi%2C%5C%3A%20q%3D%20%5Cphi%281-%5Cmu%29%29%7D
"{0 \< \\alpha, \\gamma, \\mu \< 1 ,\\;\\; \\phi \> 0, \\;\\; f_{beta}(r;\\mu,\\phi) \\sim Beta(p= \\mu\\phi,\\: q= \\phi(1-\\mu))}")

</center>

here ![\\alpha](https://ibm.codecogs.com/png.latex?%5Calpha "\\alpha")
is the probability that the response is equal to zero,
![\\gamma](https://ibm.codecogs.com/png.latex?%5Cgamma "\\gamma") is the
probability that the response is equal to one, given the probability
that the response is not equal to zero,
![\\mu](https://ibm.codecogs.com/png.latex?%5Cmu "\\mu") is the expected
value of the beta distribution,
![\\phi](https://ibm.codecogs.com/png.latex?%5Cphi "\\phi") is the
precision of the beta distribution, and
![p](https://ibm.codecogs.com/png.latex?p "p") and
![q](https://ibm.codecogs.com/png.latex?q "q") are shape parameters;
![\\mu =
\\frac{p}{p+q}](https://ibm.codecogs.com/png.latex?%5Cmu%20%3D%20%5Cfrac%7Bp%7D%7Bp%2Bq%7D
"\\mu = \\frac{p}{p+q}") and ![\\phi = p +
q](https://ibm.codecogs.com/png.latex?%5Cphi%20%3D%20p%20%2B%20q
"\\phi = p + q") (Ferrari & Cribari-Neto 2004). Ultimately, ![Y \\sim
ZOIB(y;\\alpha\_y, \\gamma\_y, \\mu\_y,
\\phi\_y)](https://ibm.codecogs.com/png.latex?Y%20%5Csim%20ZOIB%28y%3B%5Calpha_y%2C%20%5Cgamma_y%2C%20%5Cmu_y%2C%20%5Cphi_y%29
"Y \\sim ZOIB(y;\\alpha_y, \\gamma_y, \\mu_y, \\phi_y)") and ![M \\sim
ZOIB(m;\\alpha\_m, \\gamma\_m, \\mu\_m,
\\phi\_m)](https://ibm.codecogs.com/png.latex?M%20%5Csim%20ZOIB%28m%3B%5Calpha_m%2C%20%5Cgamma_m%2C%20%5Cmu_m%2C%20%5Cphi_m%29
"M \\sim ZOIB(m;\\alpha_m, \\gamma_m, \\mu_m, \\phi_m)").

The ![k^{th}](https://ibm.codecogs.com/png.latex?k%5E%7Bth%7D "k^{th}")
moment for the density ![(7)](https://ibm.codecogs.com/png.latex?%287%29
"(7)") of ![R](https://ibm.codecogs.com/png.latex?R "R") and its’
variance can be written as:

<center>

  
![\\qquad E\[R^{k}\] = (1-\\alpha)\\gamma +
(1-\\alpha)(1-\\gamma)\\mu\_k , \\quad k = 1,2,... \\qquad (7.1)\\\\
Var(R) = ((1-\\alpha)\\gamma)(1-((1-\\alpha)\\gamma)) +
(1-\\alpha)(1-\\gamma)\\left \[ \\frac{\\mu(1-\\mu)}{\\phi+1} +
\\mu^2-\\mu(1-\\alpha)(2\\gamma+\\mu(1-\\gamma)) \\right \] \\quad
(7.2)](https://ibm.codecogs.com/png.latex?%5Cqquad%20E%5BR%5E%7Bk%7D%5D%20%3D%20%281-%5Calpha%29%5Cgamma%20%2B%20%281-%5Calpha%29%281-%5Cgamma%29%5Cmu_k%20%2C%20%5Cquad%20k%20%3D%201%2C2%2C...%20%5Cqquad%20%287.1%29%5C%5C%0AVar%28R%29%20%3D%20%28%281-%5Calpha%29%5Cgamma%29%281-%28%281-%5Calpha%29%5Cgamma%29%29%20%2B%20%281-%5Calpha%29%281-%5Cgamma%29%5Cleft%20%5B%20%5Cfrac%7B%5Cmu%281-%5Cmu%29%7D%7B%5Cphi%2B1%7D%20%2B%20%5Cmu%5E2-%5Cmu%281-%5Calpha%29%282%5Cgamma%2B%5Cmu%281-%5Cgamma%29%29%20%5Cright%20%5D%20%5Cquad%20%287.2%29
"\\qquad E[R^{k}] = (1-\\alpha)\\gamma + (1-\\alpha)(1-\\gamma)\\mu_k , \\quad k = 1,2,... \\qquad (7.1)\\\\
Var(R) = ((1-\\alpha)\\gamma)(1-((1-\\alpha)\\gamma)) + (1-\\alpha)(1-\\gamma)\\left [ \\frac{\\mu(1-\\mu)}{\\phi+1} + \\mu^2-\\mu(1-\\alpha)(2\\gamma+\\mu(1-\\gamma)) \\right ] \\quad (7.2)")  

</center>

#### Bayesian approach to Causal Mediation Analysis

##### Bayesian Approach

![R \\sim f\_{ZOIB}(r;\\alpha, \\gamma, \\mu,
\\phi)](https://ibm.codecogs.com/png.latex?R%20%5Csim%20f_%7BZOIB%7D%28r%3B%5Calpha%2C%20%5Cgamma%2C%20%5Cmu%2C%20%5Cphi%29
"R \\sim f_{ZOIB}(r;\\alpha, \\gamma, \\mu, \\phi)") where we assume

<center>

  
![\\begin{align\*}h^{-1}(\\alpha) = X\\beta\_{\\alpha} \\qquad \\\\
h^{-1}(\\gamma) = X\\beta\_{\\gamma} \\qquad \\\\
h^{-1}(\\mu) = X\\beta\_{\\mu} \\qquad \\\\
exp(\\phi) = X\\beta\_{\\phi} \\qquad \\\\
\\\\
\\beta\_{\[-1\]\\alpha} \\sim MVN(0,\\Sigma\_{\\alpha}^{2}), \\\\
\\beta\_{\[-1\]\\gamma} \\sim MVN(0,\\Sigma\_{\\gamma}^{2}), \\\\
\\beta\_{\[-1\]\\mu} \\sim MVN(0,\\Sigma\_{\\mu}^{2}), \\\\
\\beta\_{\[-1\]\\phi} \\sim
MVN(0,\\Sigma\_{\\phi}^{2})\\end{align\*}](https://ibm.codecogs.com/png.latex?%5Cbegin%7Balign%2A%7Dh%5E%7B-1%7D%28%5Calpha%29%20%3D%20X%5Cbeta_%7B%5Calpha%7D%20%5Cqquad%20%5C%5C%0Ah%5E%7B-1%7D%28%5Cgamma%29%20%3D%20X%5Cbeta_%7B%5Cgamma%7D%20%5Cqquad%20%5C%5C%0Ah%5E%7B-1%7D%28%5Cmu%29%20%3D%20X%5Cbeta_%7B%5Cmu%7D%20%5Cqquad%20%5C%5C%0Aexp%28%5Cphi%29%20%3D%20X%5Cbeta_%7B%5Cphi%7D%20%5Cqquad%20%5C%5C%0A%20%5C%5C%0A%5Cbeta_%7B%5B-1%5D%5Calpha%7D%20%5Csim%20MVN%280%2C%5CSigma_%7B%5Calpha%7D%5E%7B2%7D%29%2C%20%5C%5C%0A%5Cbeta_%7B%5B-1%5D%5Cgamma%7D%20%5Csim%20MVN%280%2C%5CSigma_%7B%5Cgamma%7D%5E%7B2%7D%29%2C%20%5C%5C%0A%5Cbeta_%7B%5B-1%5D%5Cmu%7D%20%5Csim%20MVN%280%2C%5CSigma_%7B%5Cmu%7D%5E%7B2%7D%29%2C%20%5C%5C%0A%5Cbeta_%7B%5B-1%5D%5Cphi%7D%20%5Csim%20MVN%280%2C%5CSigma_%7B%5Cphi%7D%5E%7B2%7D%29%5Cend%7Balign%2A%7D
"\\begin{align*}h^{-1}(\\alpha) = X\\beta_{\\alpha} \\qquad \\\\
h^{-1}(\\gamma) = X\\beta_{\\gamma} \\qquad \\\\
h^{-1}(\\mu) = X\\beta_{\\mu} \\qquad \\\\
exp(\\phi) = X\\beta_{\\phi} \\qquad \\\\
 \\\\
\\beta_{[-1]\\alpha} \\sim MVN(0,\\Sigma_{\\alpha}^{2}), \\\\
\\beta_{[-1]\\gamma} \\sim MVN(0,\\Sigma_{\\gamma}^{2}), \\\\
\\beta_{[-1]\\mu} \\sim MVN(0,\\Sigma_{\\mu}^{2}), \\\\
\\beta_{[-1]\\phi} \\sim MVN(0,\\Sigma_{\\phi}^{2})\\end{align*}")  

</center>

where the link function ![h](https://ibm.codecogs.com/png.latex?h "h")
is a logit link function;
![h^{-1}](https://ibm.codecogs.com/png.latex?h%5E%7B-1%7D "h^{-1}"), is
the inverse of the link function that map values to a proportion between
0 and 1. ![MVN](https://ibm.codecogs.com/png.latex?MVN "MVN") is the
Multivariate Normal distribution, and
![\\beta\_{\[-1\]}](https://ibm.codecogs.com/png.latex?%5Cbeta_%7B%5B-1%5D%7D
"\\beta_{[-1]}") represents distribution of the
![\\beta](https://ibm.codecogs.com/png.latex?%5Cbeta "\\beta")
coefficients, excluding the intercept; the intercept of every
![\\beta](https://ibm.codecogs.com/png.latex?%5Cbeta "\\beta")
coefficient is assigned a uniform prior distribution.

For the mediator model, ![X](https://ibm.codecogs.com/png.latex?X "X")
is a matrix containing the intercept, the baseline covariates,
![Z](https://ibm.codecogs.com/png.latex?Z "Z"), and the treatment
variable ![A](https://ibm.codecogs.com/png.latex?A "A"). For the outcome
model, ![X](https://ibm.codecogs.com/png.latex?X "X") is a matrix
containing the intercept, the baseline covariates,
![Z](https://ibm.codecogs.com/png.latex?Z "Z"), the treatment variable
![A](https://ibm.codecogs.com/png.latex?A "A"), and the mediator
variable ![M](https://ibm.codecogs.com/png.latex?M "M") under a
specified treatment.

##### Procedure

1.  Assign the prior covariance matrix per parameter for the
    ![\\beta](https://ibm.codecogs.com/png.latex?%5Cbeta "\\beta")
    coefficients.
2.  Using these priors and the observed data, fit the models for the
    mediator and the outcome. Simulate the model parameters to obtain
    the estimated coefficients for each sampling distribution:
    ![\\theta\_M](https://ibm.codecogs.com/png.latex?%5Ctheta_M
    "\\theta_M") and
    ![\\theta\_Y](https://ibm.codecogs.com/png.latex?%5Ctheta_Y
    "\\theta_Y").
3.  For each chain of the simulation: create a new dataset, of size
    ![sim](https://ibm.codecogs.com/png.latex?sim "sim"), by taking a
    random sample, of the rows, with replacement,from the original data.
    The probabilty of selection for each row is given by
    ![\\omega\_i](https://ibm.codecogs.com/png.latex?%5Comega_i
    "\\omega_i"); ![\\omega = (\\omega\_1, \\omega\_2,...\\omega\_n)
    \\sim
    Dir(1,1,...,1)](https://ibm.codecogs.com/png.latex?%5Comega%20%3D%20%28%5Comega_1%2C%20%5Comega_2%2C...%5Comega_n%29%20%5Csim%20Dir%281%2C1%2C...%2C1%29
    "\\omega = (\\omega_1, \\omega_2,...\\omega_n) \\sim Dir(1,1,...,1)").
4.  Duplicate the dataset created in step 3. In one of the duplicated
    datasets, set the treatment variable to
    ![a](https://ibm.codecogs.com/png.latex?a "a") for the entire
    dataset. In the other duplicated dataset, set the treatment variable
    to ![a'](https://ibm.codecogs.com/png.latex?a%27 "a'").
5.  Using the duplicated datasets created in step 4, simulate new values
    of the mediator for each subject. Simulate these new mediator values
    using the estimated parameters from step 2. This step will result in
    two variables, the mediator under treatment
    ![a](https://ibm.codecogs.com/png.latex?a "a"),
    (![M^a](https://ibm.codecogs.com/png.latex?M%5Ea "M^a")), and the
    meditor under treatment
    ![a'](https://ibm.codecogs.com/png.latex?a%27 "a'"),
    (![M^{a'}](https://ibm.codecogs.com/png.latex?M%5E%7Ba%27%7D
    "M^{a'}")).
6.  Using the duplicated datasets from step 4 and the new simulated
    mediator values from step 5, create a total of four datasets. Half
    of these datasets will have the treatment variable set to
    ![a](https://ibm.codecogs.com/png.latex?a "a") for the entire
    dataset and the other half will have the treatment variable set to
    ![a'](https://ibm.codecogs.com/png.latex?a%27 "a'") for the entire
    dataset. For the half of the datasets that have treatment set to
    ![a](https://ibm.codecogs.com/png.latex?a "a"), set the mediator
    value for one entire dataset to
    ![M^a](https://ibm.codecogs.com/png.latex?M%5Ea "M^a") and for the
    other dataset set the mediator value to
    ![M^{a'}](https://ibm.codecogs.com/png.latex?M%5E%7Ba%27%7D
    "M^{a'}"). Do the same for the second half of the datasets that have
    the treatment set to ![a'](https://ibm.codecogs.com/png.latex?a%27
    "a'").
7.  Simulate new values of the outcome for each dataset created in step
    6. Simulate these new outcome values using the estimated parameters
    from step 2.
8.  Use the simulated outcome values from step 7 to compute the average
    direct, indirect, and total effect between the two outcome
    predictions under each treatment status.

The steps above will be programmed in STAN.

## Store data and input in a list to send to STAN

The STAN model accepts the following values stored in a list:

    * n - the total number of observations
    
    * np - the total number of predictors,excluding the intercept and the treatment
    
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
       np = ncol(scaled_z),
       sim = 1000,
       y = y,
       m = med,
       a = trt,
       z = scaled_z,    
       ## cov_m: prior for coefficients of the mediatior model; include treatment, do NOT include the intercept or mediator
       alpha_cov_m = diag(5, ncol(scaled_z)+1), ## == np + 1
       gamma_cov_m = diag(5, ncol(scaled_z)+1),
       mu_cov_m = diag(5, ncol(scaled_z)+1),
       phi_cov_m = diag(5, ncol(scaled_z)+1),
       ## cov_y: prior for coefficients of the outcome model; include the mediator and treatment, do not include the intercept
       alpha_cov_y = diag(5, ncol(scaled_z)+2),  ## == np + 2
       gamma_cov_y = diag(5, ncol(scaled_z)+2),
       mu_cov_y = diag(5, ncol(scaled_z)+2),
       phi_cov_y = diag(5, ncol(scaled_z)+2)
  )
```

## Stan Model

This model will
    return:

    * all_params_y - alpha, gamma, p, q for the outcome model (1:iterations,1:n,1:4)
    
    * all_params_m - alpha, gamma, p, q for the mediator model (1:iterations,1:n,1:4)
    
    * coef_mediator -  alpha, gamma, mu, phi;  coefficients for the mediator model (1:iterations,1:np,1:4)
    
    * coef_outcome -  alpha, gamma, mu, phi; coefficients for the outcome model (1:iterations,1:np+1,1:4)
    
    * tau - total effect (length = total iterations)
    
    * delta - causal effect (1:iterations, 2) where [t = 0, t = 1]
    
    * zeta - direct effect (1:iterations, 2) where [t = 0, t = 1]
    
    * pred_m - generated quantities, prediction of the mediator (1:iterations, 1:sim, 2) where [t = 0, t = 1]
    
    * pred_y - generated quantities, prediction of the outcome (1:iterations, 1:sim, 4) where [y0m0, y0m1, y1m1, y1m0]

*Note: pred\_y has the columns to represent ![Y^{A=a,
M^{A=a}}](https://ibm.codecogs.com/png.latex?Y%5E%7BA%3Da%2C%20M%5E%7BA%3Da%7D%7D
"Y^{A=a, M^{A=a}}").*

We can fit the model in Stan with the following code
.

``` r
# stan_zoib <- stan_model('~/Google Drive TheStudent/Dissertation Hours/Code/STAN/bk_zoib_regression_github.stan')
# test_fit <- sampling(stan_zoib, data = jobs_data)

stan_zoib <- stan_model(model_code = '
functions{
  matrix calc_zoib_par(matrix x_f,matrix coef_f){
    vector[rows(x_f)] p_f;
    vector[rows(x_f)] q_f;
    matrix[rows(x_f), cols(coef_f)] x_theta;
    matrix[rows(x_f), cols(coef_f)] params_hold;
    matrix[rows(x_f), 2] new_alpha_gamma;
    matrix[rows(x_f), 2] p_and_q;
      
    x_theta =  x_f * coef_f;   
      x_theta[,1] = inv_logit(x_theta[,1]);
      x_theta[,2] = inv_logit(x_theta[,2]);
      x_theta[,3] = inv_logit(x_theta[,3]);
      x_theta[,4] = exp(x_theta[,4]);
    p_f = x_theta[,3] .* x_theta[,4];
    q_f = x_theta[,4] - p_f;
  
    p_and_q = append_col(p_f,q_f);
    new_alpha_gamma = append_col(x_theta[,1], x_theta[,2]);
    params_hold = append_col(new_alpha_gamma, p_and_q); 
  return params_hold ;
  } 

  matrix calc_pred(matrix param_pred, int num_trt_splits){
      matrix[(cols(param_pred)-num_trt_splits), rows(param_pred)] wt;
      int sim = rows(param_pred);
      int d = 1;
      int i = 1;
      
      while( d < cols(param_pred)-3){
        wt[d,] = to_row_vector(param_pred[,i]);
        wt[d+1,] = to_row_vector((rep_vector(1,sim)-param_pred[,i]) .* param_pred[,i+1]);
        wt[d+2,] = to_row_vector((rep_vector(1,sim)-param_pred[,i]) .* (rep_vector(1,sim)-param_pred[,i+1]));
        d += 3;
        i += 4;
      }
  return wt;
  } 
} 
data{
  int n;
  int np; // number of parameters excluding intercept and treatment
  int sim;
  vector<lower=0, upper=1>[n] y;
  vector<lower=0, upper=1>[n] m;
  vector[n] a; //treatment variable
  matrix[n, np] z;
  matrix[np+1, np+1] alpha_cov_m;
  matrix[np+1, np+1] gamma_cov_m;
  matrix[np+1, np+1] mu_cov_m;
  matrix[np+1, np+1] phi_cov_m;
  matrix[np+2, np+2] alpha_cov_y;
  matrix[np+2, np+2] gamma_cov_y;
  matrix[np+2, np+2] mu_cov_y;
  matrix[np+2, np+2] phi_cov_y;
}
transformed data{
  matrix[n, np+2] x; //ncol(z), trt, int
  matrix[n, np+3] x_out;
  x = append_col(append_col(rep_vector(1,n),z), a); //insert col for intercept of 1s
  x_out = append_col(x, m);
}
parameters{
  matrix[np+2, 4] coef_mediator;
  matrix[np+3, 4] coef_outcome;
}
transformed parameters{
  matrix[n, 4] all_params_m;
  matrix[n, 4] all_params_y;
  all_params_m = calc_zoib_par(x, coef_mediator);
  all_params_y = calc_zoib_par(x_out, coef_outcome);
}
model{
  // coefficients for mediator model; does not include the mediator
  coef_mediator[2:,1] ~ multi_normal(rep_vector(0,cols(x)-1), alpha_cov_m);
  coef_mediator[2:,2] ~ multi_normal(rep_vector(0,cols(x)-1), gamma_cov_m);
  coef_mediator[2:,3] ~ multi_normal(rep_vector(0,cols(x)-1), mu_cov_m);
  coef_mediator[2:,4] ~ multi_normal(rep_vector(0,cols(x)-1), phi_cov_m);
  
  // coefficients for outcome model; includes the mediator
  coef_outcome[2:,1] ~ multi_normal(rep_vector(0,cols(x_out)-1), alpha_cov_y);
  coef_outcome[2:,2] ~ multi_normal(rep_vector(0,cols(x_out)-1), gamma_cov_y);
  coef_outcome[2:,3] ~ multi_normal(rep_vector(0,cols(x_out)-1), mu_cov_y);
  coef_outcome[2:,4] ~ multi_normal(rep_vector(0,cols(x_out)-1), phi_cov_y);

  // zero one inflated beta likelihood
    for (i in 1:n) {
      if (y[i] == 0) {
        target += log(all_params_y[i,1]) ;
      } else if (y[i] == 1) {
        target += log1m(all_params_y[i,1]) + log(all_params_y[i,2]);
      } else {
        target += log1m(all_params_y[i,1]) + log1m(all_params_y[i,2]) + beta_lpdf(y[i] | all_params_y[i,3], all_params_y[i,4]);
      }
    
      if (m[i] == 0) {
        target += log(all_params_m[i,1]);
      } else if (m[i] == 1) {
        target += log1m(all_params_m[i,1]) + log(all_params_m[i,2]);
      } else {
        target += log1m(all_params_m[i,1]) + log1m(all_params_m[i,2]) + beta_lpdf(m[i] | all_params_m[i,3], all_params_m[i,4]);
      }
    }
}
generated quantities{
  real tau;
  vector[2] delta;
  vector[2] zeta;
  matrix[sim, 2] pred_m;
  matrix[sim, 4] pred_y;
  {
    int index;
    matrix[sim, 16] param_pred_y;
    matrix[sim, 8] param_pred_m;
    vector[rows(z)] wt;
    matrix[6, sim] wt_m; // three possible outcomes for density per trt
    matrix[12, sim] wt_y;
    matrix[sim, np+2] X_sample;
    matrix[sim, np+2] X_m0;
    matrix[sim, np+2] X_m1;
    matrix[sim, np+3] X_y0_m0;
    matrix[sim, np+3] X_y0_m1;
    matrix[sim, np+3] X_y1_m0;
    matrix[sim, np+3] X_y1_m1;
    
      wt = dirichlet_rng(rep_vector(1, rows(z))); 
      for (j in 1:sim){
        index = categorical_rng(wt);
        X_sample[j,:] = x[index,:];
      }

      X_m0 = X_sample;
      X_m0[:,cols(x)] = rep_vector(0, sim); // cols(x) == index for last col of x, which is treat

      X_m1 = X_sample;
      X_m1[:,cols(x)] = rep_vector(1, sim);
      
      // calculate new alpha, gamma, mu, phi
      param_pred_m[,1:4] = calc_zoib_par(X_m0, coef_mediator);
      param_pred_m[,5:8] = calc_zoib_par(X_m1, coef_mediator);
    
      wt_m = calc_pred(param_pred_m, 2);

      for(k in 1:sim){
        int index_wtm0 = categorical_rng(wt_m[1:3,k]);
        int index_wtm1 = categorical_rng(wt_m[4:6,k]);
        
        if (index_wtm0 == 1){ pred_m[k,1] = 0;}
          else if (index_wtm0 == 2){pred_m[k,1] = 1;}
            else if (index_wtm0 == 3){pred_m[k,1] = beta_rng(param_pred_m[k,3],param_pred_m[k,4]);}
            
        if (index_wtm1 == 1){ pred_m[k,2] = 0;}
          else if (index_wtm1 == 2){pred_m[k,2] = 1;}
            else if (index_wtm1 == 3){pred_m[k,2] = beta_rng(param_pred_m[k,7],param_pred_m[k,8]);}
      }
      
      X_y0_m0 = append_col(X_m0, pred_m[,1]);
      X_y0_m1 = append_col(X_m0, pred_m[,2]);
      X_y1_m1 = append_col(X_m1, pred_m[,2]);
      X_y1_m0 = append_col(X_m1, pred_m[,1]);
      
      //coef_mediator is np+2 x 4. it includes the mediator
      param_pred_y[,1:4] = calc_zoib_par(X_y0_m0, coef_outcome);
      param_pred_y[,5:8] = calc_zoib_par(X_y0_m1, coef_outcome);
      param_pred_y[,9:12] = calc_zoib_par(X_y1_m1, coef_outcome);
      param_pred_y[,13:16] = calc_zoib_par(X_y1_m0, coef_outcome);
      
      wt_y = calc_pred(param_pred_y, 4);
      
      for(h in 1:sim){
          int index_y0m0 = categorical_rng(wt_y[1:3,h]);
          int index_y0m1 = categorical_rng(wt_y[4:6,h]);
          int index_y1m1 = categorical_rng(wt_y[7:9,h]);
          int index_y1m0 = categorical_rng(wt_y[10:12,h]);

          if (index_y0m0 == 1){pred_y[h,1] = 0;}
          else if (index_y0m0 == 2){pred_y[h,1] = 1;}
          else if (index_y0m0 == 3){pred_y[h,1] = beta_rng(param_pred_y[h,3],param_pred_y[h,4]);}

          if (index_y0m1 == 1){ pred_y[h,2] = 0;}
          else if (index_y0m1 == 2){pred_y[h,2] = 1;}
          else if (index_y0m1 == 3){pred_y[h,2] = beta_rng(param_pred_y[h,7],param_pred_y[h,8]);}

          if (index_y1m1 == 1){ pred_y[h,3] = 0;}
          else if (index_y1m1 == 2){pred_y[h,3] = 1;}
          else if (index_y1m1 == 3){pred_y[h,3] = beta_rng(param_pred_y[h,11],param_pred_y[h,12]);}

          if (index_y1m0 == 1){ pred_y[h,4] = 0;}
          else if (index_y1m0 == 2){pred_y[h,4] = 1;}
          else if (index_y1m0 == 3){pred_y[h,4] = beta_rng(param_pred_y[h,15],param_pred_y[h,16]);}
      }
      
      delta[1] = mean(pred_y[:,2]) - mean(pred_y[:,1]);
      delta[2] = mean(pred_y[:,3]) - mean(pred_y[:,4]);
      zeta[1] = mean(pred_y[:,4]) - mean(pred_y[:,1]);
      zeta[2] = mean(pred_y[:,3]) - mean(pred_y[:,2]);
      tau = mean(pred_y[:,3]) - mean(pred_y[:,1]);

  }  // end of local variables
}  //end of generated quantities

')

test_fit <- sampling(stan_zoib, data = jobs_data, seed = num_seed)
```

## Assessing STAN output

When assessing STAN output, one of the things that you want to check is
whether the chains are converging, and that they are converging to the
same area. Some of the recommended convergence checks include monitoring
the potential scale reduction (PSR) factor, ![\\hat
R](https://ibm.codecogs.com/png.latex?%5Chat%20R "\\hat R") statistic,
and using visual checks, e.g. traceplots. ![\\hat
R](https://ibm.codecogs.com/png.latex?%5Chat%20R "\\hat R") evaluates
the mixing of the chains by comparing the variation between the chains
to the variation within the chains. “The condition of ![\\hat
R](https://ibm.codecogs.com/png.latex?%5Chat%20R "\\hat R") being ‘near’
1 depends on the problem at hand, but we generally have been satisfied
with setting 1.1 as a threshold” (Gelman et al., 2004).

Additionally, since the Markov Chain Monte Carlo (MCMC) does not return
independent draws, the simulations within each chain will show some
level of autocorrelation. This autocorrelation inceases the uncertainty
of the estimation of poesterior quantities. The amount by which this
autocrorrelation increases in estimates can be measured by the effective
sample size (ESS),
![n\_{eff}](https://ibm.codecogs.com/png.latex?n_%7Beff%7D "n_{eff}"),
which should be large so that it can provide a measure of precision;
![n\_{eff}](https://ibm.codecogs.com/png.latex?n_%7Beff%7D "n_{eff}") is
the ‘effective number of independent simulation draws’.

    ## $coef_mediator
    ## [1] 4000    5    4
    ## 
    ## $coef_outcome
    ## [1] 4000    6    4
    ## 
    ## $all_params_m
    ## [1] 4000  899    4
    ## 
    ## $all_params_y
    ## [1] 4000  899    4
    ## 
    ## $tau
    ## [1] 4000
    ## 
    ## $delta
    ## [1] 4000    2
    ## 
    ## $zeta
    ## [1] 4000    2
    ## 
    ## $pred_m
    ## [1] 4000 1000    2
    ## 
    ## $pred_y
    ## [1] 4000 1000    4
    ## 
    ## $lp__
    ## [1] 4000

### Assess Rhat and Effective Sample Size

<table class="table table-striped table-hover" style="margin-left: auto; margin-right: auto;">

<caption>

Table 1

</caption>

<thead>

<tr>

<th style="text-align:left;">

</th>

<th style="text-align:left;">

Min.

</th>

<th style="text-align:left;">

1st Qu.

</th>

<th style="text-align:left;">

Median

</th>

<th style="text-align:left;">

Mean

</th>

<th style="text-align:left;">

3rd
Qu.

</th>

<th style="text-align:left;">

Max.

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

r\_hat

</td>

<td style="text-align:left;">

0.9990

</td>

<td style="text-align:left;">

0.9996

</td>

<td style="text-align:left;">

0.9999

</td>

<td style="text-align:left;">

1.0000

</td>

<td style="text-align:left;">

1.0003

</td>

<td style="text-align:left;">

1.0028

</td>

</tr>

<tr>

<td style="text-align:left;">

n\_eff

</td>

<td style="text-align:left;">

1879

</td>

<td style="text-align:left;">

3794

</td>

<td style="text-align:left;">

3987

</td>

<td style="text-align:left;">

4085

</td>

<td style="text-align:left;">

4268

</td>

<td style="text-align:left;">

6188

</td>

</tr>

</tbody>

</table>

### Assess Traceplots

![](README_files/figure-gfm/asses_stan_traceplots-1.png)<!-- -->![](README_files/figure-gfm/asses_stan_traceplots-2.png)<!-- -->

## Summarizing the Posteriors

<br />

<div style="border: 1px solid #ddd; padding: 0px; overflow-y: scroll; height:300px; overflow-x: scroll; width:800px; ">

<table class="table table-striped table-hover table-condensed" style="margin-left: auto; margin-right: auto;">

<caption>

Table
3

</caption>

<thead>

<tr>

<th style="text-align:center;position: sticky; top:0; background-color: #FFFFFF;">

param

</th>

<th style="text-align:center;position: sticky; top:0; background-color: #FFFFFF;">

specified\_param

</th>

<th style="text-align:center;position: sticky; top:0; background-color: #FFFFFF;">

overall\_param

</th>

<th style="text-align:center;position: sticky; top:0; background-color: #FFFFFF;">

mean

</th>

<th style="text-align:center;position: sticky; top:0; background-color: #FFFFFF;">

se\_mean

</th>

<th style="text-align:center;position: sticky; top:0; background-color: #FFFFFF;">

sd

</th>

<th style="text-align:center;position: sticky; top:0; background-color: #FFFFFF;">

2.5%

</th>

<th style="text-align:center;position: sticky; top:0; background-color: #FFFFFF;">

97.5%

</th>

<th style="text-align:center;position: sticky; top:0; background-color: #FFFFFF;">

n\_eff

</th>

<th style="text-align:center;position: sticky; top:0; background-color: #FFFFFF;">

Rhat

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:center;font-weight: bold;vertical-align: top !important;" rowspan="20">

coef\_mediator

</td>

<td style="text-align:center;">

alpha

</td>

<td style="text-align:center;">

coef\_mediator\[1,1\]

</td>

<td style="text-align:center;">

\-8.9504

</td>

<td style="text-align:center;">

0.0375

</td>

<td style="text-align:center;">

1.8388

</td>

<td style="text-align:center;">

\-13.0256

</td>

<td style="text-align:center;">

\-5.8110

</td>

<td style="text-align:center;">

2407.206

</td>

<td style="text-align:center;">

1.0004

</td>

</tr>

<tr>

<td style="text-align:center;">

gamma

</td>

<td style="text-align:center;">

coef\_mediator\[1,2\]

</td>

<td style="text-align:center;">

\-2.0734

</td>

<td style="text-align:center;">

0.0036

</td>

<td style="text-align:center;">

0.1845

</td>

<td style="text-align:center;">

\-2.4544

</td>

<td style="text-align:center;">

\-1.7168

</td>

<td style="text-align:center;">

2633.013

</td>

<td style="text-align:center;">

0.9998

</td>

</tr>

<tr>

<td style="text-align:center;">

mu

</td>

<td style="text-align:center;">

coef\_mediator\[1,3\]

</td>

<td style="text-align:center;">

0.9339

</td>

<td style="text-align:center;">

0.0009

</td>

<td style="text-align:center;">

0.0469

</td>

<td style="text-align:center;">

0.8415

</td>

<td style="text-align:center;">

1.0235

</td>

<td style="text-align:center;">

2881.332

</td>

<td style="text-align:center;">

1.0000

</td>

</tr>

<tr>

<td style="text-align:center;">

phi

</td>

<td style="text-align:center;">

coef\_mediator\[1,4\]

</td>

<td style="text-align:center;">

1.9972

</td>

<td style="text-align:center;">

0.0016

</td>

<td style="text-align:center;">

0.0818

</td>

<td style="text-align:center;">

1.8351

</td>

<td style="text-align:center;">

2.1497

</td>

<td style="text-align:center;">

2761.153

</td>

<td style="text-align:center;">

0.9999

</td>

</tr>

<tr>

<td style="text-align:center;">

alpha

</td>

<td style="text-align:center;">

coef\_mediator\[2,1\]

</td>

<td style="text-align:center;">

\-0.7935

</td>

<td style="text-align:center;">

0.0130

</td>

<td style="text-align:center;">

0.8101

</td>

<td style="text-align:center;">

\-2.4931

</td>

<td style="text-align:center;">

0.7039

</td>

<td style="text-align:center;">

3863.642

</td>

<td style="text-align:center;">

0.9993

</td>

</tr>

<tr>

<td style="text-align:center;">

gamma

</td>

<td style="text-align:center;">

coef\_mediator\[2,2\]

</td>

<td style="text-align:center;">

0.2248

</td>

<td style="text-align:center;">

0.0014

</td>

<td style="text-align:center;">

0.0971

</td>

<td style="text-align:center;">

0.0303

</td>

<td style="text-align:center;">

0.4160

</td>

<td style="text-align:center;">

4805.615

</td>

<td style="text-align:center;">

0.9993

</td>

</tr>

<tr>

<td style="text-align:center;">

mu

</td>

<td style="text-align:center;">

coef\_mediator\[2,3\]

</td>

<td style="text-align:center;">

0.0215

</td>

<td style="text-align:center;">

0.0004

</td>

<td style="text-align:center;">

0.0286

</td>

<td style="text-align:center;">

\-0.0335

</td>

<td style="text-align:center;">

0.0764

</td>

<td style="text-align:center;">

4564.880

</td>

<td style="text-align:center;">

1.0004

</td>

</tr>

<tr>

<td style="text-align:center;">

phi

</td>

<td style="text-align:center;">

coef\_mediator\[2,4\]

</td>

<td style="text-align:center;">

\-0.0105

</td>

<td style="text-align:center;">

0.0007

</td>

<td style="text-align:center;">

0.0495

</td>

<td style="text-align:center;">

\-0.1094

</td>

<td style="text-align:center;">

0.0841

</td>

<td style="text-align:center;">

5237.077

</td>

<td style="text-align:center;">

1.0003

</td>

</tr>

<tr>

<td style="text-align:center;">

alpha

</td>

<td style="text-align:center;">

coef\_mediator\[3,1\]

</td>

<td style="text-align:center;">

\-0.1578

</td>

<td style="text-align:center;">

0.0123

</td>

<td style="text-align:center;">

0.8158

</td>

<td style="text-align:center;">

\-1.8231

</td>

<td style="text-align:center;">

1.4481

</td>

<td style="text-align:center;">

4382.138

</td>

<td style="text-align:center;">

1.0003

</td>

</tr>

<tr>

<td style="text-align:center;">

gamma

</td>

<td style="text-align:center;">

coef\_mediator\[3,2\]

</td>

<td style="text-align:center;">

0.1074

</td>

<td style="text-align:center;">

0.0014

</td>

<td style="text-align:center;">

0.0945

</td>

<td style="text-align:center;">

\-0.0779

</td>

<td style="text-align:center;">

0.2944

</td>

<td style="text-align:center;">

4732.798

</td>

<td style="text-align:center;">

1.0002

</td>

</tr>

<tr>

<td style="text-align:center;">

mu

</td>

<td style="text-align:center;">

coef\_mediator\[3,3\]

</td>

<td style="text-align:center;">

\-0.0408

</td>

<td style="text-align:center;">

0.0004

</td>

<td style="text-align:center;">

0.0283

</td>

<td style="text-align:center;">

\-0.0956

</td>

<td style="text-align:center;">

0.0151

</td>

<td style="text-align:center;">

5060.141

</td>

<td style="text-align:center;">

0.9995

</td>

</tr>

<tr>

<td style="text-align:center;">

phi

</td>

<td style="text-align:center;">

coef\_mediator\[3,4\]

</td>

<td style="text-align:center;">

\-0.1245

</td>

<td style="text-align:center;">

0.0008

</td>

<td style="text-align:center;">

0.0497

</td>

<td style="text-align:center;">

\-0.2225

</td>

<td style="text-align:center;">

\-0.0273

</td>

<td style="text-align:center;">

4114.331

</td>

<td style="text-align:center;">

1.0013

</td>

</tr>

<tr>

<td style="text-align:center;">

alpha

</td>

<td style="text-align:center;">

coef\_mediator\[4,1\]

</td>

<td style="text-align:center;">

0.3589

</td>

<td style="text-align:center;">

0.0106

</td>

<td style="text-align:center;">

0.6949

</td>

<td style="text-align:center;">

\-1.0403

</td>

<td style="text-align:center;">

1.7257

</td>

<td style="text-align:center;">

4287.051

</td>

<td style="text-align:center;">

1.0003

</td>

</tr>

<tr>

<td style="text-align:center;">

gamma

</td>

<td style="text-align:center;">

coef\_mediator\[4,2\]

</td>

<td style="text-align:center;">

\-0.0775

</td>

<td style="text-align:center;">

0.0014

</td>

<td style="text-align:center;">

0.0956

</td>

<td style="text-align:center;">

\-0.2649

</td>

<td style="text-align:center;">

0.1095

</td>

<td style="text-align:center;">

4791.682

</td>

<td style="text-align:center;">

0.9999

</td>

</tr>

<tr>

<td style="text-align:center;">

mu

</td>

<td style="text-align:center;">

coef\_mediator\[4,3\]

</td>

<td style="text-align:center;">

0.0887

</td>

<td style="text-align:center;">

0.0004

</td>

<td style="text-align:center;">

0.0275

</td>

<td style="text-align:center;">

0.0340

</td>

<td style="text-align:center;">

0.1408

</td>

<td style="text-align:center;">

4470.367

</td>

<td style="text-align:center;">

0.9996

</td>

</tr>

<tr>

<td style="text-align:center;">

phi

</td>

<td style="text-align:center;">

coef\_mediator\[4,4\]

</td>

<td style="text-align:center;">

0.1232

</td>

<td style="text-align:center;">

0.0007

</td>

<td style="text-align:center;">

0.0495

</td>

<td style="text-align:center;">

0.0267

</td>

<td style="text-align:center;">

0.2213

</td>

<td style="text-align:center;">

4730.354

</td>

<td style="text-align:center;">

0.9996

</td>

</tr>

<tr>

<td style="text-align:center;">

alpha

</td>

<td style="text-align:center;">

coef\_mediator\[5,1\]

</td>

<td style="text-align:center;">

1.6897

</td>

<td style="text-align:center;">

0.0297

</td>

<td style="text-align:center;">

1.6135

</td>

<td style="text-align:center;">

\-1.2281

</td>

<td style="text-align:center;">

5.0271

</td>

<td style="text-align:center;">

2947.680

</td>

<td style="text-align:center;">

1.0009

</td>

</tr>

<tr>

<td style="text-align:center;">

gamma

</td>

<td style="text-align:center;">

coef\_mediator\[5,2\]

</td>

<td style="text-align:center;">

0.3637

</td>

<td style="text-align:center;">

0.0042

</td>

<td style="text-align:center;">

0.2136

</td>

<td style="text-align:center;">

\-0.0488

</td>

<td style="text-align:center;">

0.8020

</td>

<td style="text-align:center;">

2648.469

</td>

<td style="text-align:center;">

0.9999

</td>

</tr>

<tr>

<td style="text-align:center;">

mu

</td>

<td style="text-align:center;">

coef\_mediator\[5,3\]

</td>

<td style="text-align:center;">

0.0157

</td>

<td style="text-align:center;">

0.0011

</td>

<td style="text-align:center;">

0.0585

</td>

<td style="text-align:center;">

\-0.0998

</td>

<td style="text-align:center;">

0.1300

</td>

<td style="text-align:center;">

2885.014

</td>

<td style="text-align:center;">

1.0002

</td>

</tr>

<tr>

<td style="text-align:center;">

phi

</td>

<td style="text-align:center;">

coef\_mediator\[5,4\]

</td>

<td style="text-align:center;">

\-0.1123

</td>

<td style="text-align:center;">

0.0019

</td>

<td style="text-align:center;">

0.1007

</td>

<td style="text-align:center;">

\-0.3045

</td>

<td style="text-align:center;">

0.0826

</td>

<td style="text-align:center;">

2887.499

</td>

<td style="text-align:center;">

1.0002

</td>

</tr>

<tr>

<td style="text-align:center;font-weight: bold;vertical-align: top !important;" rowspan="24">

coef\_outcome

</td>

<td style="text-align:center;">

alpha

</td>

<td style="text-align:center;">

coef\_outcome\[1,1\]

</td>

<td style="text-align:center;">

\-5.1968

</td>

<td style="text-align:center;">

0.0124

</td>

<td style="text-align:center;">

0.6308

</td>

<td style="text-align:center;">

\-6.4650

</td>

<td style="text-align:center;">

\-3.9728

</td>

<td style="text-align:center;">

2572.552

</td>

<td style="text-align:center;">

1.0017

</td>

</tr>

<tr>

<td style="text-align:center;">

gamma

</td>

<td style="text-align:center;">

coef\_outcome\[1,2\]

</td>

<td style="text-align:center;">

\-10.1844

</td>

<td style="text-align:center;">

0.0726

</td>

<td style="text-align:center;">

3.1484

</td>

<td style="text-align:center;">

\-16.9582

</td>

<td style="text-align:center;">

\-4.6876

</td>

<td style="text-align:center;">

1878.556

</td>

<td style="text-align:center;">

1.0009

</td>

</tr>

<tr>

<td style="text-align:center;">

mu

</td>

<td style="text-align:center;">

coef\_outcome\[1,3\]

</td>

<td style="text-align:center;">

\-0.4544

</td>

<td style="text-align:center;">

0.0028

</td>

<td style="text-align:center;">

0.1382

</td>

<td style="text-align:center;">

\-0.7191

</td>

<td style="text-align:center;">

\-0.1815

</td>

<td style="text-align:center;">

2388.798

</td>

<td style="text-align:center;">

1.0001

</td>

</tr>

<tr>

<td style="text-align:center;">

phi

</td>

<td style="text-align:center;">

coef\_outcome\[1,4\]

</td>

<td style="text-align:center;">

1.8013

</td>

<td style="text-align:center;">

0.0048

</td>

<td style="text-align:center;">

0.2256

</td>

<td style="text-align:center;">

1.3648

</td>

<td style="text-align:center;">

2.2367

</td>

<td style="text-align:center;">

2173.031

</td>

<td style="text-align:center;">

1.0002

</td>

</tr>

<tr>

<td style="text-align:center;">

alpha

</td>

<td style="text-align:center;">

coef\_outcome\[2,1\]

</td>

<td style="text-align:center;">

\-0.3482

</td>

<td style="text-align:center;">

0.0017

</td>

<td style="text-align:center;">

0.1129

</td>

<td style="text-align:center;">

\-0.5664

</td>

<td style="text-align:center;">

\-0.1307

</td>

<td style="text-align:center;">

4502.747

</td>

<td style="text-align:center;">

0.9993

</td>

</tr>

<tr>

<td style="text-align:center;">

gamma

</td>

<td style="text-align:center;">

coef\_outcome\[2,2\]

</td>

<td style="text-align:center;">

2.0742

</td>

<td style="text-align:center;">

0.0266

</td>

<td style="text-align:center;">

1.3449

</td>

<td style="text-align:center;">

\-0.1962

</td>

<td style="text-align:center;">

5.1265

</td>

<td style="text-align:center;">

2548.968

</td>

<td style="text-align:center;">

1.0002

</td>

</tr>

<tr>

<td style="text-align:center;">

mu

</td>

<td style="text-align:center;">

coef\_outcome\[2,3\]

</td>

<td style="text-align:center;">

0.2046

</td>

<td style="text-align:center;">

0.0004

</td>

<td style="text-align:center;">

0.0320

</td>

<td style="text-align:center;">

0.1420

</td>

<td style="text-align:center;">

0.2677

</td>

<td style="text-align:center;">

5167.922

</td>

<td style="text-align:center;">

1.0001

</td>

</tr>

<tr>

<td style="text-align:center;">

phi

</td>

<td style="text-align:center;">

coef\_outcome\[2,4\]

</td>

<td style="text-align:center;">

\-0.1274

</td>

<td style="text-align:center;">

0.0008

</td>

<td style="text-align:center;">

0.0517

</td>

<td style="text-align:center;">

\-0.2285

</td>

<td style="text-align:center;">

\-0.0268

</td>

<td style="text-align:center;">

3902.050

</td>

<td style="text-align:center;">

1.0006

</td>

</tr>

<tr>

<td style="text-align:center;">

alpha

</td>

<td style="text-align:center;">

coef\_outcome\[3,1\]

</td>

<td style="text-align:center;">

\-0.1161

</td>

<td style="text-align:center;">

0.0016

</td>

<td style="text-align:center;">

0.1122

</td>

<td style="text-align:center;">

\-0.3333

</td>

<td style="text-align:center;">

0.1059

</td>

<td style="text-align:center;">

4775.536

</td>

<td style="text-align:center;">

0.9997

</td>

</tr>

<tr>

<td style="text-align:center;">

gamma

</td>

<td style="text-align:center;">

coef\_outcome\[3,2\]

</td>

<td style="text-align:center;">

1.5747

</td>

<td style="text-align:center;">

0.0260

</td>

<td style="text-align:center;">

1.5324

</td>

<td style="text-align:center;">

\-0.9448

</td>

<td style="text-align:center;">

4.9056

</td>

<td style="text-align:center;">

3468.863

</td>

<td style="text-align:center;">

1.0009

</td>

</tr>

<tr>

<td style="text-align:center;">

mu

</td>

<td style="text-align:center;">

coef\_outcome\[3,3\]

</td>

<td style="text-align:center;">

0.0751

</td>

<td style="text-align:center;">

0.0005

</td>

<td style="text-align:center;">

0.0312

</td>

<td style="text-align:center;">

0.0137

</td>

<td style="text-align:center;">

0.1361

</td>

<td style="text-align:center;">

4210.932

</td>

<td style="text-align:center;">

0.9999

</td>

</tr>

<tr>

<td style="text-align:center;">

phi

</td>

<td style="text-align:center;">

coef\_outcome\[3,4\]

</td>

<td style="text-align:center;">

\-0.0956

</td>

<td style="text-align:center;">

0.0008

</td>

<td style="text-align:center;">

0.0487

</td>

<td style="text-align:center;">

\-0.1946

</td>

<td style="text-align:center;">

0.0007

</td>

<td style="text-align:center;">

4157.896

</td>

<td style="text-align:center;">

0.9997

</td>

</tr>

<tr>

<td style="text-align:center;">

alpha

</td>

<td style="text-align:center;">

coef\_outcome\[4,1\]

</td>

<td style="text-align:center;">

0.1249

</td>

<td style="text-align:center;">

0.0015

</td>

<td style="text-align:center;">

0.1071

</td>

<td style="text-align:center;">

\-0.0826

</td>

<td style="text-align:center;">

0.3341

</td>

<td style="text-align:center;">

5426.100

</td>

<td style="text-align:center;">

0.9998

</td>

</tr>

<tr>

<td style="text-align:center;">

gamma

</td>

<td style="text-align:center;">

coef\_outcome\[4,2\]

</td>

<td style="text-align:center;">

\-0.2966

</td>

<td style="text-align:center;">

0.0168

</td>

<td style="text-align:center;">

1.1281

</td>

<td style="text-align:center;">

\-2.6803

</td>

<td style="text-align:center;">

1.7931

</td>

<td style="text-align:center;">

4485.079

</td>

<td style="text-align:center;">

1.0005

</td>

</tr>

<tr>

<td style="text-align:center;">

mu

</td>

<td style="text-align:center;">

coef\_outcome\[4,3\]

</td>

<td style="text-align:center;">

0.0239

</td>

<td style="text-align:center;">

0.0005

</td>

<td style="text-align:center;">

0.0317

</td>

<td style="text-align:center;">

\-0.0366

</td>

<td style="text-align:center;">

0.0878

</td>

<td style="text-align:center;">

3661.124

</td>

<td style="text-align:center;">

1.0003

</td>

</tr>

<tr>

<td style="text-align:center;">

phi

</td>

<td style="text-align:center;">

coef\_outcome\[4,4\]

</td>

<td style="text-align:center;">

0.0009

</td>

<td style="text-align:center;">

0.0008

</td>

<td style="text-align:center;">

0.0499

</td>

<td style="text-align:center;">

\-0.0967

</td>

<td style="text-align:center;">

0.0991

</td>

<td style="text-align:center;">

4018.184

</td>

<td style="text-align:center;">

1.0013

</td>

</tr>

<tr>

<td style="text-align:center;">

alpha

</td>

<td style="text-align:center;">

coef\_outcome\[5,1\]

</td>

<td style="text-align:center;">

\-0.1199

</td>

<td style="text-align:center;">

0.0031

</td>

<td style="text-align:center;">

0.2251

</td>

<td style="text-align:center;">

\-0.5540

</td>

<td style="text-align:center;">

0.3293

</td>

<td style="text-align:center;">

5226.583

</td>

<td style="text-align:center;">

1.0003

</td>

</tr>

<tr>

<td style="text-align:center;">

gamma

</td>

<td style="text-align:center;">

coef\_outcome\[5,2\]

</td>

<td style="text-align:center;">

\-1.7458

</td>

<td style="text-align:center;">

0.0238

</td>

<td style="text-align:center;">

1.6422

</td>

<td style="text-align:center;">

\-5.1164

</td>

<td style="text-align:center;">

1.4574

</td>

<td style="text-align:center;">

4756.706

</td>

<td style="text-align:center;">

0.9995

</td>

</tr>

<tr>

<td style="text-align:center;">

mu

</td>

<td style="text-align:center;">

coef\_outcome\[5,3\]

</td>

<td style="text-align:center;">

\-0.0639

</td>

<td style="text-align:center;">

0.0010

</td>

<td style="text-align:center;">

0.0637

</td>

<td style="text-align:center;">

\-0.1889

</td>

<td style="text-align:center;">

0.0598

</td>

<td style="text-align:center;">

3913.214

</td>

<td style="text-align:center;">

1.0002

</td>

</tr>

<tr>

<td style="text-align:center;">

phi

</td>

<td style="text-align:center;">

coef\_outcome\[5,4\]

</td>

<td style="text-align:center;">

\-0.0411

</td>

<td style="text-align:center;">

0.0015

</td>

<td style="text-align:center;">

0.1041

</td>

<td style="text-align:center;">

\-0.2469

</td>

<td style="text-align:center;">

0.1640

</td>

<td style="text-align:center;">

4850.796

</td>

<td style="text-align:center;">

0.9992

</td>

</tr>

<tr>

<td style="text-align:center;">

alpha

</td>

<td style="text-align:center;">

coef\_outcome\[6,1\]

</td>

<td style="text-align:center;">

3.9055

</td>

<td style="text-align:center;">

0.0142

</td>

<td style="text-align:center;">

0.7342

</td>

<td style="text-align:center;">

2.4803

</td>

<td style="text-align:center;">

5.3786

</td>

<td style="text-align:center;">

2670.785

</td>

<td style="text-align:center;">

1.0013

</td>

</tr>

<tr>

<td style="text-align:center;">

gamma

</td>

<td style="text-align:center;">

coef\_outcome\[6,2\]

</td>

<td style="text-align:center;">

\-0.3808

</td>

<td style="text-align:center;">

0.0324

</td>

<td style="text-align:center;">

2.0556

</td>

<td style="text-align:center;">

\-4.3820

</td>

<td style="text-align:center;">

3.6692

</td>

<td style="text-align:center;">

4015.854

</td>

<td style="text-align:center;">

0.9997

</td>

</tr>

<tr>

<td style="text-align:center;">

mu

</td>

<td style="text-align:center;">

coef\_outcome\[6,3\]

</td>

<td style="text-align:center;">

\-1.1036

</td>

<td style="text-align:center;">

0.0036

</td>

<td style="text-align:center;">

0.1774

</td>

<td style="text-align:center;">

\-1.4529

</td>

<td style="text-align:center;">

\-0.7666

</td>

<td style="text-align:center;">

2407.988

</td>

<td style="text-align:center;">

1.0003

</td>

</tr>

<tr>

<td style="text-align:center;">

phi

</td>

<td style="text-align:center;">

coef\_outcome\[6,4\]

</td>

<td style="text-align:center;">

0.2175

</td>

<td style="text-align:center;">

0.0061

</td>

<td style="text-align:center;">

0.2898

</td>

<td style="text-align:center;">

\-0.3520

</td>

<td style="text-align:center;">

0.7805

</td>

<td style="text-align:center;">

2238.491

</td>

<td style="text-align:center;">

1.0002

</td>

</tr>

<tr>

<td style="text-align:center;font-weight: bold;">

tau

</td>

<td style="text-align:center;vertical-align: top !important;" rowspan="5">

causal
effects

</td>

<td style="text-align:center;">

tau

</td>

<td style="text-align:center;">

\-0.0115

</td>

<td style="text-align:center;">

0.0002

</td>

<td style="text-align:center;">

0.0129

</td>

<td style="text-align:center;">

\-0.0370

</td>

<td style="text-align:center;">

0.0128

</td>

<td style="text-align:center;">

4215.443

</td>

<td style="text-align:center;">

0.9998

</td>

</tr>

<tr>

<td style="text-align:center;font-weight: bold;vertical-align: top !important;" rowspan="2">

delta

</td>

<td style="text-align:center;">

delta\[1\]

</td>

<td style="text-align:center;">

\-0.0032

</td>

<td style="text-align:center;">

0.0001

</td>

<td style="text-align:center;">

0.0079

</td>

<td style="text-align:center;">

\-0.0188

</td>

<td style="text-align:center;">

0.0123

</td>

<td style="text-align:center;">

3995.480

</td>

<td style="text-align:center;">

1.0001

</td>

</tr>

<tr>

<td style="text-align:center;">

delta\[2\]

</td>

<td style="text-align:center;">

\-0.0028

</td>

<td style="text-align:center;">

0.0001

</td>

<td style="text-align:center;">

0.0077

</td>

<td style="text-align:center;">

\-0.0180

</td>

<td style="text-align:center;">

0.0124

</td>

<td style="text-align:center;">

3755.008

</td>

<td style="text-align:center;">

1.0007

</td>

</tr>

<tr>

<td style="text-align:center;font-weight: bold;vertical-align: top !important;" rowspan="2">

zeta

</td>

<td style="text-align:center;">

zeta\[1\]

</td>

<td style="text-align:center;">

\-0.0087

</td>

<td style="text-align:center;">

0.0002

</td>

<td style="text-align:center;">

0.0125

</td>

<td style="text-align:center;">

\-0.0332

</td>

<td style="text-align:center;">

0.0161

</td>

<td style="text-align:center;">

4166.851

</td>

<td style="text-align:center;">

0.9997

</td>

</tr>

<tr>

<td style="text-align:center;">

zeta\[2\]

</td>

<td style="text-align:center;">

\-0.0083

</td>

<td style="text-align:center;">

0.0002

</td>

<td style="text-align:center;">

0.0123

</td>

<td style="text-align:center;">

\-0.0331

</td>

<td style="text-align:center;">

0.0158

</td>

<td style="text-align:center;">

4170.490

</td>

<td style="text-align:center;">

0.9998

</td>

</tr>

</tbody>

</table>

</div>

<br />
![](README_files/figure-gfm/visualize_posterior_graphs-1.png)<!-- -->![](README_files/figure-gfm/visualize_posterior_graphs-2.png)<!-- -->![](README_files/figure-gfm/visualize_posterior_graphs-3.png)<!-- -->

<br />

##### References

  - Ferrari, S., and Cribari-Neto, F. (2004). “Beta regression for
    modelling rates and proportions,” Journal of Applied Statistics,
    31(7), 799-815.
  - Gelman, A., Carlin, J.B., Stern, H.S., and Rubin, D.B. (2004). 
    Bayesian data analysis (2nd ed.)
  - Imai, K., Keele, L., and Tingley, D. (2010),“A General Approach
    to Causal Mediation Analysis,”Psychological Methods, 15(4),
    309–334.
