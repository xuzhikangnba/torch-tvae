# Objective and scope

Let the latent dimension be d. The generative model is

\[
u \sim \mathrm{Gamma}(\nu/2,\nu/2),\qquad
z\mid u \sim \mathcal N(m, S/u),\qquad
x\mid z \sim p_\theta(x\mid z),
\]

where Gamma uses **shape and rate**, and S = diag(scale squared).
There is one scalar u per latent vector, shared across all its coordinates.
Integrating u gives an elliptical multivariate Student-t, not a product of
independent univariate Student-t distributions. For nu > 2, its covariance is
nu/(nu-2) times S; S itself is a scale matrix, not the marginal covariance.

The encoder returns q(z|x) = N(mu, diag(exp(logvar))). In the variational family
q(z,u|x) = q(z|x) q(u|x), define

\[
\Delta = \sum_j \frac{\exp(\ell_j)+(\mu_j-m_j)^2}{s_j^2}.
\]

The optimal auxiliary factor **within this family** is

\[
q^*(u\mid x)=\mathrm{Gamma}(a,b),\qquad
a=(\nu+d)/2,\quad b=(\nu+\Delta)/2.
\]

This is not the exact posterior of the entire nonlinear generative model.
It uses encoder moments; it is not q(u|z) evaluated at one sampled z.

The library's regularizer is the joint KL

\[
R = \mathbb E_{q(u)}\left[\mathrm{KL}(q(z|x)\|p(z|u))\right]
  + \mathrm{KL}(q(u|x)\|p(u)).
\]

Both terms are analytic. With E[u] = a/b and E[log u] = digamma(a) - log b,

\[
R_N=\tfrac12\left(E[u]\Delta-dE[\log u]
       +2\sum_j\log s_j-\sum_j\ell_j-d\right),
\]

\[
R_G=(a-a_0)\psi(a)+\log\Gamma(a_0)-\log\Gamma(a)
   +a_0(\log b-\log b_0)+a(b_0/b-1),\quad a_0=b_0=\nu/2.
\]

The chain rule gives

\[
R = \mathrm{KL}(q(z|x)\|p_t(z))
 + \mathbb E_{q(z|x)}\mathrm{KL}(q(u|x)\|p(u|z))
 \geq \mathrm{KL}(q(z|x)\|p_t(z)).
\]

Consequently the API is named `kl_bound`, not an exact marginal Gaussian-to-t
KL. The nonlinear reconstruction expectation is estimated with a
reparameterized Gaussian latent sample. No Monte Carlo sample of u is needed
for training. Sampling u is used only when drawing Student-t prior samples.

## Relation to the paper implementation

The reference t-VaDE code has K components and adds a categorical KL. This
library's first version uses one component: the categorical term vanishes,
there are no cluster assignments, and no GMM refit or prior overwrite is run.
The prior location and scale default to fixed zero and one; learning them is
optional. Degrees of freedom can be fixed or learned.

The paper reference trainer sums reconstruction over the batch and averages
the prior penalty over the batch. This library first sums event coordinates
for each sample and then averages both terms over the batch. Therefore its
beta is independent of batch size. The old coefficient 5.0 must not be carried
over unchanged as if the two objectives were identical: at batch size B the
reference relative coefficient corresponds to 5/B after dividing its total
loss by B. The new default is beta=1.

This package implements the underlying construction and a new basic t-VAE
interface. It is not an unchanged wrapper around the paper's experiment
scripts, and it does not claim to reproduce multi-component clustering
tables. Keep exact experiment reproduction in a separately versioned release.

## Numerical conventions

- Encoder logvar is log variance; prior log_scale is log standard scale.
- df = softplus(raw_df) + min_df, with min_df=0.001 by default.
- No extra epsilon is added to encoder variance while retaining the original
  log variance in the entropy; those quantities remain consistent.
- Float32 and float64 are supported. Large degrees of freedom involve
  cancellation in special functions; use float64 for Gaussian-limit studies.
- The prior can have infinite marginal variance when df <= 2. Its density and
  analytic variational regularizer remain defined; empirical variance is not
  a valid diagnostic in that regime.
