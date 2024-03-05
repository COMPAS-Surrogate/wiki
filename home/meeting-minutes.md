# ⏲️ Meeting Minutes

## March XX, 2024

#### LnL nans

Likelihood equation: $$\log L(D|\lambda) = \log L_{\rm Poisson}(N_{obs}|\lambda) + \sum^{N_{obs}} \log p(D_i | \lambda)$$

Poisson LnL: $$\log L_{\rm Poisson}(N_{obs}|\lambda) \propto N_{obs} \log(N_{model}(\lambda))-N_{model}(\lambda)$$

(Note -- we ignore terms that depend on the data only. These disappear on normalisation, such as log(Nobs!) and permutation coefficients.)

Grid LnL: $$p(D_i | \lambda) = p(z=z_i, M_c=M_c,i|\lambda)$$

My bug was in the `Grid LnL` --> Im getting Nans when using \~1000 BBHs and even \~10 BBHs:

|                                                                                                                                  |                                                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| ![plot\_n16\_bbh\_population](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/ad25dd2f-e7d9-428f-afcd-265c578dfaff) | ![plot\_n1372\_bbh\_population](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/2b256a77-3ccd-4685-aef0-126d5efb5483) |

`(aSF=0.01, dSF=4.70, mu_z=-0.01, sigma0=0.0)` ![plot\_cosmo\_0 010\_2 770\_2 900\_4 700\_0 035\_-0 230](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/7b7bfb06-54f4-43b0-ad0a-3223ac2564b2)

Priors

```
    muz=[-0.5, -0.001],  # Jeff's alpha
    sigma0=[0.1, 0.6],  # Jeff's sigma
    aSF=[0.005, 0.015],
    dSF=[4.2, 5.2],

```

<details>

<summary>To reproduce nan:</summary>

```python
    lnl, unc = McZGrid.lnl(
        mcz_obs=mock_data.observations.mcz,
        duration=1,
        compas_h5_path=mock_data.compas_filename,
        sf_sample=dict(aSF=0.01, dSF=4.70, mu_z=-0.01, sigma0=0.0),
        n_bootstraps=0,
        save_plots=True,
        outdir=f"{tmp_path}/nan_lnl"
    )
```

</details>

<details>

<summary>Code for section with Nan:</summary>

```python


def ln_poisson_likelihood(
    n_obs: int, n_model: int, ignore_factorial=True
) -> float:
    """
    Computes LnL(N_obs | N_model) = N_obs * ln(N_model) - N_model - ln(N_obs!)

    :param n_obs: number of observed events
    :param n_model: number of events predicted by the model
    :param ignore_factorial: ignore the factorial term in the likelihood

    # TODO: Why are we ignoring the factorial term?? It was in Ilya's notes from 2023, but unsure why...

    :return: the log likelihood
    """
    if n_model <= 0:
        return -np.inf
    lnl = n_obs * np.log(n_model) - n_model

    if ignore_factorial is False:
        lnl += -np.log(np.math.factorial(n_obs))

    return lnl






def ln_mcz_grid_likelihood(
    mcz_obs: np.ndarray, model_prob_func: Callable
) -> float:
    """
    Computes LnL(mc, z | model) = sum_i  ln p(mc_i, z_i | model)     (for N_obs events)
    :param mcz_obs: [[mc,z], [mc,z], ...] Array of observed mc and z values for each event (exact measurement)
    :param model_prob_func: model_func(mc,z) -> prob(mc_i, z_i | model)
    :return:
    """
    return np.sum([np.log(model_prob_func(mc, z)) for mc, z in mcz_obs])


...


    def prob_of_mcz(self, mc: float, z: float, duration: float = 1.0) -> float:
        mc_bin, z_bin = self.get_matrix_bin_idx(mc, z)
        return self.rate_matrix[mc_bin, z_bin] / self.n_detections(duration)




# FINALLY: 
lnl = poisson_lnl + mcz_lnl

```

The `self.n_detections(duration)`--> 0 leading to the nan

</details>

is my Current fix valid: `prob_of_mcz`--> 0 if $N\_{model}==0$?

#### Multimodal 1d Gaussian





**Basic test**

{% embed url="https://compas-surrogate.github.io/lnl_surrogate/studies/example.html#bimodal-gaussian" %}
Bimodal example
{% endembed %}



**Posterior generation**

**PP-Plot**

## Feb 19th, 2024

**Meeting with Jeff, Ilya, Chayan, Avi**

### Todos:

* Multimodal lnl distribution -- one bigger, one smaller peak (start at smaller peak), see if the different acquisition functions reach the higher peak (sharper peak but wider).
* difference between current and Maximum lnl off all points sampled (don't know ground truth, but we do know max LnL amongst all points)
* everything relative to MaxLnL
* LnL Nans: maybe because of small numbers --> Maybe I have lots of observations -- fewer detections will help avoid the nans -- right now we are using N-bbh = 1K, will this get fixed temporarily with N-bbh = 10s?
* Test generation of Posteriors (with 1D multimodal thing)
  * Use MCMC + surrogate LnL
    * When using uncertainty: If LnL: 25 +/- 2 (then we draw LnL from `Norm(25, sigma=2)`)
    * When not using uncertainty: just take the surrogate Mean
  * compare analytical with surrogate posterior
  * make a PP-plot,
* Diagnostic plot:
  * When we have analytical: KL div between true posterior (analytical) + surrogate Posterior
  * For COMPAS: KL div between _best_ posterior (ie after 10000 iterations) + current posterior

### Summary

* [Example of 'minimum' LnL training points and model predictions over iterations](https://compas-surrogate.github.io/lnl\_surrogate/README.html) ![](https://raw.githubusercontent.com/COMPAS-Surrogate/lnl\_surrogate/main/docs/studies/regret.png)
* [1D sf example](https://compas-surrogate.github.io/pipeline/studies/1D/1D.html) The 'errors' are much larger... hmm
* [2D sf example](https://github.com/COMPAS-Surrogate/pipeline/blob/main/scripts/two\_dim/bo\_v2.py)... sorry, forgot to include model predictions! ![lnl\_error](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/7ef87183-c62f-4c45-b23c-5990d91256af)

Params are `aSF, dSF` ![round\_round3](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/e86bad5b-fad4-4684-8913-5e92ff0c660a)

### Questions:

* Posterior from LnL -- do I need to sample, or just `LnL * prior`?
* using ilya's analytical Unc -- checked with group, for now this is OK (for production we'll need to move to bootstrapping)
* stopping criteria for BayesOpt?

## Jan 31st, 2024

**Meeting with Jeff, Ilya, Chayan, Avi**

### NExt steps:

* Assume COMPAS run producing the 'correct' universe
* making regret plot with both surrogate and tru LnL
* can we zoom into main region -- meaningful posteriors?

### different threads for this project:

1. how good of a LnL estimate do we need? What is the correct tradeoff between making COMPAS calls (1000 points in parameter space with high fidelity), and surrogate modeling
2. Best way to create an acquisition function?

Jeff: How do we choose best 'x' for acquisition -- can we use covarience

Avi: Do we really need a LnL surrogate or more just an active learning method to find 'interesting' regions of parameter space?

|                                                                                                                            |                                                                                                                     |                                                                                                                     |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| ![bo\_metrics\_round99](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/b37d60ab-f764-4cf5-8455-1069b61a2589) | ![eval\_round99](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/5951b00b-36ec-42ab-b4ca-89a765ea5902) | ![func\_round99](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/8106be2c-7080-4e3f-9bd7-e536ba61459c) |

Ilya:

* I think we ultimately want to zoom in, a lot. A difference of even 100 in ln likelihood, which is about the most I can resolve on this plot, means these parameters are 10^{43} times more/less likely -- that's huge, and would mean the parameters in question are far outside any credible interval of interest.
* Are lnL NANs because you are multiplying / dividing by numbers that are too large / small? Sometimes it's a question of offsetting things relative to some suitable fixed value rather than computing absolute likelihoods.

![Screenshot 2024-01-31 at 10 50 09 am](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/30750f50-09d4-4e78-9552-b37aaa543b8b)

### Ilya's notes on LnL uncertainty:

```
What is the expected level of fluctuation in the log likelihood due to statistical sampling errors?
We can think of the likelihood as a product of likelihoods of the observed count of events c_i given Poisson predictions n_i for the count of events in each bin i in Mchirp-redshift space.  Thus, the log likelihood is the sum of log likelihoods over each bin.  The Poisson likelihood in each bin is

L_i = exp(-n_i) n_i^c_i / c_i!

Ignoring the denominator (which depends only on data, and is thus a normalisation constant),
log L_i = -n_i + c_i log n_i.

Then the total log likelihood is
log L = - \sum_i n_i + \sum_i c_i log n_i.

Suppose there is an error d_i in the predicted counts in each bin due to sampling uncertainties.  Assume d_i << n_i; then
log (n_i+d_i) = log(n_i) log(1+d_i/n_i) ~ log(n_i) (1+ d_i/n_i - (1/2) (d_i/n_i)^2 + ...).

The associated error in log L will then be
d log L ~ - \sum_i d_i + \sum_i c_i (d_i/n_i) log n_i + ... = - \sum_i d_i + \sum_i d_i (c_i/n_i) log n_i + ...

The first term is easy: it's just the fluctuation in the total number of expected detections due to finite sampling accuracy.
For a given i, the second term can generally be larger than the first term, i.e., (c_i/n_i) log n_i can be larger than 1.  However, the total contribution of the first term will generally be of the same order or greater than the second term, because of the stochastic nature of the second term (the number of observations in some bins exceeds expectations while there are too few observations in other bins).
To see this, let's include some assumptions which aren't necessary but which will simplify the argument, and consider two extreme regimes.  The additional assumptions are that d_i and n_i are the same in all K bins, i.e., n_i=N/K and d_i = D/K. Let's further suppose we are evaluating the likelihood at the true model parameters, so that all c_i are independently drawn from a Poisson distribution with expectation value N/K.   The two regimes we will consider are (a) n_i = N/K ~ a few and (b) n_i = N/K \ll 1.  In both regimes, the first term sums up to D.

In regime (a), we expect that c_i have a scatter of \sqrt{N/K} around the expectation value of N/K.  Typically, sqrt(K) of the elements in the second term will be too large (or too small). The second term will then have a typical value of sqrt(K) (D/K) log(N/K).   Thus, the second term is of order the first term times log(N/K) / sqrt(K).  As long as the number of bins is sufficiently large that sqrt(K) > log(N/K), the first term dominates or sets the expected size of the fluctuation in log L.
In regime (b), we expect that the bins are sufficiently finely pixellated so that the number of expected detections in any one bin is small, n_i = N/K \ll 1.  Then, c_i should all be either zero or 1.  When they are zero, the pixels contribute nothing to the second term.  The total number of pixels we expect to contribute is ~N.  The second term is thus N ((D/K) / (N/K)) log (N/K) ~ D log(N/K).  While the second term can then be a factor of |log(N/K)| larger than the first term, this is only a logarithmic scaling.  Moreover, it is really the effective K (the number of pixels containing the majority of the expectation value) that should be used here.  Thus, as long as the effective K is not too many orders of magnitude larger than N, it's still safe to say that the fluctuation in log L from statistical sampling uncertainty is no more than a few times (log(N/Keff), to be precise) larger than the statistical uncertainty in the total number of expected detections.
```

## Jan 10, 2024

### Some notes on exploration strategies and acquisition functions

$x\_{\rm next} = {\rm GetQuery}(model, data) = {\rm arg max} \alpha(x; model, data)$

Where is the next point that maximizes the future value of the model??

[check out this lecture](https://youtu.be/C5nqEHpdyoE?si=zIE3ZJffk3lukYkA\&t=2781)

|                                     |                                 |                                                                                      |
| ----------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| Sample(model, x)                    | Thompson Sampling               | Least likely to be stuck at locals, sample GP-posterior, maximumise sampled function |
| GetTail(model, x, threshold)        | PI (probability of improvement) | Greedy -- local optima, not best for high dimensions, or multimodal distributions    |
| GetImprovement(model, x, threshold) | EI (expected improvement)/EGO   | Expectation instead of probability, considered more global than PI                   |
| GetQuantile(model, x, quantile)     | UCB (upper confidence bound)    | The acquisition function 'envelops' max uncertainty/quantile at a region             |
| GetEntropy(model, x)                | PES (predictive entropy search) |                                                                                      |

The acquisition function needs to be selected/configured to balance the following depending on your task:

* EXPLOIT: Maximize function f
* EXPLORE: "LEARN" function f

![Screenshot 2024-01-04 at 3 33 04 pm](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/690af4a4-b914-4b82-a835-1360876015c0)

#### Paper that studies switching acquisition functions:

Check out [PI is back! Switching Acquisition Functions in Bayesian Optimization](https://arxiv.org/pdf/2211.01455.pdf)

They switch between different acquisition functions -- what we want to do!

They use:

* EI -- explorative AF
* PI -- exploitative AF

And state:

> if the properties of the problem to optimize are unknown, switching from EI to PI after 25 % of the budget of surrogate-based evaluations should be favored.

Related papers:

* [Self-Adjusting Weighted Expected Improvement for Bayesian Optimization](https://arxiv.org/pdf/2306.04262.pdf)
* [Mastering the exploration-exploitation trade-off in Bayesian Optimization](https://arxiv.org/abs/2305.08624)

![ei\_pi](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/3f72bb5b-533c-4269-91c3-2622be01d40f) ![pi](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/59c3eeea-b799-4ac9-816a-f438181b0652) ![ei](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/35559cc3-7fc4-4e59-b769-539fd7d72282) ![2d](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/79e0f6b5-f7af-437c-8c5e-6185fadfecb6)

**'Regret' plot**

This is a 'regret' plot looking at different acquisition functions. The red line marks the LnL when using the 'true' parameters. ![regret](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/29081402-e576-40a6-a845-4ebb10c7d67d)

## Dec 20, 2023

* Chayan, Jeff, Avi chat about project + bring Chayan up to speed
* [Slides from COMPAS group meeting](https://docs.google.com/presentation/d/1qQsZRgjjdnA3l-NnkThWMDxUIcRqqTkz7JjWKe4j9p8/edit?usp=sharing)

**Action items**

* Add chayan to https://github.com/COMPAS-Surrogate (need Chayan's username etc)
* chayan to investigate TFprobability, etc as surrogate options
* Avi to make an example of the

## Dec 15, 2023

* Avi's main progress Dec 12-14:
  * Refactor LnL computation
  * slurm utils for pipeline
  * faster acquisition function test cases [(see this test)](https://github.com/COMPAS-Surrogate/pipeline/blob/main/tests/test\_aSF\_1d\_pipeline.py)
* QUESTIONS:
  * Multiple dimension acquisition function?
  * Go over params to be used--> \["aSF", "dSF", "mu\_z", "sigma\_z"]??
    * https://www.praescientem.com.au/publications/Riley\_2023\_ApJ\_950\_80.pdf
    * it should be mu0 sigma0
  * Ask about using posteriors vs list of 'delta' functions of Mc-Z mock observations
  * Ilya's suggestion for faster bootstraps?
  * code design for acquisition functions for generalized models? Where should this code live?
* Remaining TODOs:
  * DNN model for \[Sf params] --> \[Lnl, unc] (output both)
  * testing out multi-D acquisition function...

## Nov 2, 2023

Things to discuss:

* COMPAS surrogate results
* GP with uncertain inputs
* Acquisition function

### Code overview

All repos here: https://github.com/COMPAS-Surrogate/

* LnL computer:
  * Generates training data (runs cosmic-int, saves matrices, computes LnL)
  * Generate matrices with `make_detection_matricies --aSF 0.1 --dSf ...`
  * Or `make_detection_matricies --param_csv data.csv`
  * After, `compute_lnl --detection-matrices matrices.hdf --data universe.hdf`
  * **QS: Maybe just save LnL, don't bother saving matrices?**
  * **TODO: add in slurm generator**
* LnL surrogate:
  * Contains surrogate models (GPFlow, SklearnGP, SklearnMLP, TfDNN)
  * Contains acquisition-functions (not working for more than 1D atm)
* Pipeline:
  * Will contain slurm pipeline builder for N repetitions of `Compute + Surrogate + acquisition`
  * Currently contains bunch of python manual testing workflows

### COMPAS surrogate results

* Without incorporating LnL uncertainty, results "look" ok, but the validation error is high (overfitting) ![](https://avivajpeyi.github.io/compas\_al\_expts/\_images/sampling\_summary.png)
* LnL + unc surrogate has a high training error (even with some random points)
* **Something is going wrong in the GP surrogate model.**

### GP with uncertain inputs

Currently testing with `GPFlow` (a 'deep' GP) and `sklearn.GP`

* GPs perform well when not using uncertain inputs (LnL unc), and perform well in 1D (with uncertain inputs).
* GPs not performing well with 2+D with uncertain inputs...

#### QS

1. What about a DNN with two outputs (lnl + lnl unc) ?
2. What about bootstrapping the surrogate outputs (train N surrogates with different data, combine results + get LnL unc)
3. What about adding a "Noise" kernel to the GP? Do we expect the noise to be Gaussian? _NO NEED FOR BOOTSTRAPS_\_??

| Name                                | Image                                                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| gpflow\_quadratic\_uncFalse\_n20    | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/89ea204f-a264-4052-9300-9bb1d4645489) |
| gpflow\_quadratic\_uncFalse\_n50    | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/f53d0e91-d6e3-459f-8d0d-aef7b26de19b) |
| gpflow\_quadratic\_uncTrue\_n20     | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/0b2592b0-4fba-400b-ac24-2188975e9edb) |
| gpflow\_quadratic\_uncTrue\_n50     | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/9ffc37e3-5c4d-469b-b24e-707efb95f60c) |
| sklearngp\_quadratic\_uncFalse\_n10 | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/323f0314-48d8-4a7f-83e8-d75f4926cc86) |
| sklearngp\_quadratic\_uncFalse\_n20 | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/7de0802d-6a6a-4a53-8717-5a4747cad7fa) |
| sklearngp\_quadratic\_uncFalse\_n50 | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/5288cc0a-b8af-4f60-80e4-898132ad7bf4) |
| sklearngp\_quadratic\_uncTrue\_n10  | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/06e43236-799b-466c-82b0-926db2929e5d) |
| sklearngp\_quadratic\_uncTrue\_n20  | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/5ed213c3-09fd-4202-b747-763266bc5742) |
| sklearngp\_quadratic\_uncTrue\_n50  | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/190db3ab-8bba-44cb-b48e-bdac55e54ab3) |
| gpflow\_n10                         | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/018f9aee-920e-4f08-9f62-9fdd2f66fc31) |
| gpflow\_n25                         | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/73a183fd-27e1-4740-95b6-5636dba45b50) |
| gpflow\_n50                         | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/da49c8ae-6980-4d1d-b9a7-157370c24f96) |
| sklearngp\_n10                      | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/5b47f8e4-63fb-472c-a998-a3c3d100f8d0) |
| sklearngp\_n25                      | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/b7920b37-f715-4381-a07b-f0b8b956b818) |
| sklearngp\_n50                      | ![Image](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/8e3aae40-670f-4851-a1f0-b3e0dbf04d1c) |

### Acquisition functions:

Instead of coding our [own](https://avivajpeyi.github.io/compas\_al\_expts/al\_and\_gps.html), maybe we can leverage pre-existing BO tools? (note: some of these require a GP to be trained using an MCMC)

* https://modal-python.readthedocs.io/en/latest/content/examples/active\_regression.html?highlight=regression%20multiple%20dimension#Active-regression
* https://gpax.readthedocs.io/en/latest/examples.html
*   https://docs.jaxgaussianprocesses.com/examples/bayesian\_optimisation/#a-more-challenging-example-the-six-hump-camel-function

    ![Screen Shot 2023-11-02 at 12 22 07 pm](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/7425ebae-895a-4e1d-ab3c-a550171dd551)

![Screen Shot 2023-11-02 at 12 21 58 pm](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/06f9724b-ac78-4d2c-9098-033da4c5cfaf)

## Oct 3, 2023

### Summary of project:

There are 4 parts to this project:

1. **COMPAS-specific questions:** how to compute likelihoods and uncertainties on them which arise because of how COMPAS computes those likelihoods, how large do simulations need to be in order to achieve sufficiently low uncertainties;
2. **How to build surrogate models** for the log-likelihood question over the COMPAS parameter space;
3. **How to do active learning** if we want to iteratively improve the log-likelihood surrogate model;
4. **Using the surrogate model to infer the COMPAS parameters from LVK Data** -- while that's sufficiently trivial with a good likelihood model in hand, there's a potential question of whether/how to incorporate the surrogate model uncertainty into posterior distributions on the parameters. However, let's leave that for now; simple version is to draw likelihoods for given parameters from the surrogate model including its uncertainty.

**ILYA'S NOTES**

#### Notes on uncertainty

On 1, Ilya did a crude analytical estimate that suggests that the COMPAS-sampling uncertainty in log-likelihood is of the order of the sampling uncertainty in the number of predicted events (detections), as long we are near max log-likelihood. Avi and Jeff will check that; if true, it should help us understand how many samples we'll need.

![Screenshot from 2023-12-20 14-01-13](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/f1b6e2a7-562a-40b5-95c4-bef1c8eee081)

![Screenshot from 2023-12-20 13-59-17](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/20d81f35-9c9f-4be5-b080-0950ff8aa460)

![Screenshot from 2023-12-20 13-59-32](https://github.com/COMPAS-Surrogate/pipeline/assets/15642823/71e7aea5-79f4-45d2-af75-d02a02370f4a)

TAKEAWAY: Bootstraps are always relatively "cheap" because they don't require actual reruns of the full analysis (this will be particularly obvious when we go from varying post-processing MSSF parameters to varying COMPAS C++ model parameters). But, in general, the fractional accuracy of the bootstrap uncertainty estimate scales as 1/sqrt{N\_boostraps}. **So 100 bootstraps will give you a 10% fractional accuracy of the uncertainty in the quantity of interest due to finite sampling, which should indeed be good enough for our purposes.**

#### Notes on surrogate

On 2, Avi tried a few things -- scikit-learn, GPflow, GPJax. We can use any off-the-shelf tool if it's good enough, doesn't need to be perfect. Avi seems to like GPJax, so that should be fine. By the way, I'd also suggest just using point estimates for the mean and standard deviation of the GP predictor for the log-likelihood at any location in the COMPAS parameter space (if doing inference on GP hyper-parameters, as GPJax does, can just pick MAP values).

#### Notes on acquisition

On 3, the question is really what is the best acquisition function. A proposal based on our conversations and some literature is to use the following:

75% of the time (can play with the fraction), use A = exp (-(mu - mu\_max)^2/2 sigma^2), where mu and sigma are the mean and standard deviation of the GP predictor for log likelihood, and mu\_max is its max value; 25% of the time, use A = sigma The idea is that most of the time, we are only interested in improving the surrogate model in places where the likelihood has a decent chance of being higher than the currently known maximum; but sometime, to avoid getting stuck, we just want to make more evaluations in places where the surrogate model is particularly uncertain.

Also, I'm thinking that we might want to think of (normalised) A as a probability distribution for drawing the next location for a COMPAS evaluation rather than just using the parameters where A is maximal, again to avoid getting stuck, and perhaps to better prepare for the situation when we'll want to draw the next 100 parameters for evaluation rather than going through 1 step at a time (more efficient in terms of wall time when we have 100 cores available). Though in that case, we really should think of how to penalise the cost function for proposed parameter space location x\_{i+1} based on the new proposals x\_1...x\_i, which isn't accounted for here.

## Sept 5, 2023

Number of DCOs from N Compas runs -- Poisson distributions.

Currently for bootstrapping

```
for n trials:
    n_dcos[I] = N original + Normal( sqrt(N))  
```

Illya's suggestion:

* We have 1 COMPAS run with 'k' DCOs.
* Draw x from Poisson(mean=k)

```
bootstrap_lnl_vals = zeros(n_bootstraps)

for i in n_bootstraps:
   n_dcos= Poisson(lambda=l).sample() 
   compas_seeds = original_compas_population_seeds.sample(n_dcos)
   bootstrap_lnl_vals[i] = likelihood(data, compas_seeds)

unc = std(lnl_vals)
lnl = likelihood(data, original_compas_seeds)
```

1. Tests with LnL unc Ilya predicts uncertainty in LnL \~ uncertainty in N detections. Uncertainty in LnL will increase with more detections.

Tests for increasing number of bootstraps

* simulate population
* get LnL + 1000 bootstrap Lnl
* see what the Std(bootstrap lnl) is for different number of bootstraps
* also store the number of detections for all bootstrap events

2. GPU LnL?

## Aug 23, 2023

|                                                                                                                     |                                                                                                                     |                                                                                                                      |                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| ![d1\_ei](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/2cda4d1b-6cb0-4ce5-96b5-48d9cbf0790c) | ![d1\_th](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/97bc210f-c29c-4b12-8c7f-a3bb8b5abd47) | ![d1\_ucb](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/e1943785-7ea4-4079-99d3-29abb2041a1d) | ![d1\_ue](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/9f6a16cb-0be3-4a0c-8472-53e30e3bcc63) |
| ![d2\_ei](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/7ef729fc-8f95-40cb-b1a2-e1171eceec43) | ![d2\_th](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/7e9a36a5-d4d6-4593-a9d3-2ce706ccb5dc) | ![d2\_ucb](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/42e89e23-1e5b-44e0-a78b-259f52869c16) | ![d2\_ue](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/bd35eded-89c0-4f5c-a080-04010230e463) |
| ![d3\_ei](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/da7ad5b4-820b-4ce9-9680-1cf7cd50ed4f) | ![d3\_th](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/aa33ad19-e249-44d8-b5ee-f628a12266a8) | ![d3\_ucb](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/d6184215-0e05-4388-88bb-82228ea6b200) | ![d3\_ue](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/2691afbf-76d5-4168-b6e5-d5a210f31a2e) |

|     | D1                                                                                                                   | D2                                                                                                                   | D3                                                                                                                   |
| --- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| EI  | ![d1\_ei](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/2cda4d1b-6cb0-4ce5-96b5-48d9cbf0790c)  | ![d2\_ei](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/7ef729fc-8f95-40cb-b1a2-e1171eceec43)  | ![d3\_ei](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/da7ad5b4-820b-4ce9-9680-1cf7cd50ed4f)  |
| TH  | ![d1\_th](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/97bc210f-c29c-4b12-8c7f-a3bb8b5abd47)  | ![d2\_th](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/7e9a36a5-d4d6-4593-a9d3-2ce706ccb5dc)  | ![d3\_th](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/aa33ad19-e249-44d8-b5ee-f628a12266a8)  |
| UCB | ![d1\_ucb](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/e1943785-7ea4-4079-99d3-29abb2041a1d) | ![d2\_ucb](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/42e89e23-1e5b-44e0-a78b-259f52869c16) | ![d3\_ucb](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/d6184215-0e05-4388-88bb-82228ea6b200) |
| UE  | ![d1\_ue](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/9f6a16cb-0be3-4a0c-8472-53e30e3bcc63)  | ![d2\_ue](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/bd35eded-89c0-4f5c-a080-04010230e463)  | ![d3\_ue](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/2691afbf-76d5-4168-b6e5-d5a210f31a2e)  |

```



`![dataset_1_ei](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/2cda4d1b-6cb0-4ce5-96b5-48d9cbf0790c)`
`![dataset_1_thompson](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/97bc210f-c29c-4b12-8c7f-a3bb8b5abd47)`
`![dataset_1_ucb](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/e1943785-7ea4-4079-99d3-29abb2041a1d)`
`![dataset_1_ue](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/9f6a16cb-0be3-4a0c-8472-53e30e3bcc63)`


`![dataset_2_ei](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/7ef729fc-8f95-40cb-b1a2-e1171eceec43)`
`![dataset_2_thompson](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/7e9a36a5-d4d6-4593-a9d3-2ce706ccb5dc)`
`![dataset_2_ucb](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/42e89e23-1e5b-44e0-a78b-259f52869c16)`
`![dataset_2_ue](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/bd35eded-89c0-4f5c-a080-04010230e463)`




`![dataset_3_ei](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/da7ad5b4-820b-4ce9-9680-1cf7cd50ed4f)`
`![dataset_3_thompson](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/aa33ad19-e249-44d8-b5ee-f628a12266a8)`
`![dataset_3_ucb](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/d6184215-0e05-4388-88bb-82228ea6b200)`
`![dataset_3_ue](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/2691afbf-76d5-4168-b6e5-d5a210f31a2e)`



![d1_goodqUCB](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/2be19245-50e4-482a-8979-e171ae33103b)
![d1_badqUCB](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/cb9a7e54-c71b-4d4d-8229-387f2f53450e)
![d1_mse](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/c1858c15-2219-40f5-975f-228e329f3f78)
![d2_mse](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/881896f6-42e4-4ab9-b56c-c69190a60443)
![d3_mse](https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/58cb8d2c-315e-4276-ae2c-77a030655686)


```

## Aug 13, 2023

### Papers on Active learning

#### Methods/Algos

1. uncertainty-based sampling: least confident ([Lewis and Catlett](https://www.sciencedirect.com/science/article/pii/B978155860335650026X?via%3Dihub)), max margin and max entropy
2. committee-based algorithms: vote entropy, consensus entropy and max disagreement ([Cohn et al.](http://www.cs.northwestern.edu/\~pardo/courses/mmml/papers/active\_learning/improving\_generalization\_with\_active\_learning\_ML94.pdf))
3. multilabel strategies: SVM binary minimum ([Brinker](https://link.springer.com/chapter/10.1007%2F3-540-31314-1\_24)), max loss, mean max loss, ([Li et al.](http://dx.doi.org/10.1109/ICIP.2004.1421535)) MinConfidence, MeanConfidence, MinScore, MeanScore ([Esuli and Sebastiani](http://dx.doi.org/10.1007/978-3-642-00958-7\_12))
4. expected error reduction: binary and log loss ([Roy and McCallum](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.588.5666\&rep=rep1\&type=pdf))
5. Bayesian optimization: probability of improvement, expected improvement and upper confidence bound ([Snoek et al.](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf))
6. batch active learning: ranked batch-mode sampling ([Cardoso et al.](https://www.sciencedirect.com/science/article/pii/S0020025516313949))
7. information density framework ([McCallum and Nigam](http://www.kamalnigam.com/papers/emactive-icml98.pdf))
8. stream-based sampling ([Atlas et al.](https://papers.nips.cc/paper/261-training-connectionist-networks-with-queries-and-selective-sampling.pdf))
9. active regression with max standard deviance sampling for Gaussian processes or ensemble regressors

#### How to estimate uncertainty in Deep Learning networks

* [Excellent tutorial from AGW on Bayesian Deep Learning](https://icml.cc/virtual/2020/tutorial/5750)
* This tut is inspired by his publication [Bayesian Deep Learning and a Probabilistic Perspective of Generalization](https://arxiv.org/abs/2002.08791)
* [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf) (Gal and Ghahramani, 2016) This describes Monte-Carlo Dropout, a way to estimate uncertainty through stochastic dropout at test time
* [Bayesian Uncertainty Estimation for Batch Normalized Deep Networks](https://arxiv.org/abs/1802.06455) (Teye et al. 2018) This describes Monte-Carlo BatchNorm, a way to estimate uncertainty through random batch norm parameters at test time
* [Bayesian Deep Learning and a Probabilistic Perspective of Generalization](https://arxiv.org/abs/2002.08791) (Gordon Wilson and Izmailov, 2020) Presentation of multi-SWAG a mix between VI and Ensembles.
* [Advances in Variational inference](https://arxiv.org/pdf/1711.05597.pdf) (Zhang et al, 2018) Gives a quick introduction to VI and the most recent advances.
* [A Simple Baseline for Bayesian Uncertainty in Deep Learning](https://arxiv.org/abs/1902.02476) (Maddox et al. 2019) Presents SWAG, an easy way to create ensembles.

#### Bayesian active learning

* [Deep Bayesian Active Learning with Image Data](https://arxiv.org/pdf/1703.02910.pdf) (Gal and Islam and Ghahramani, 2017) Fundamental paper on how to do Bayesian active learning.
* [Sampling bias in active learning](http://cseweb.ucsd.edu/\~dasgupta/papers/twoface.pdf) (Dasgupta 2009) Presents sampling bias and how to solve it by combining heuristics and random selection.
* [Bayesian Active Learning for Classification and Preference Learning](https://arxiv.org/pdf/1112.5745.pdf) (Houlsby et al. 2011) Fundamental paper on one of the main heuristic BALD.

#### GP and active learning

* [Exploration of lattice Hamiltonians for functional and structural discovery via Gaussian process-based exploration–exploitation](https://pubs.aip.org/aip/jap/article/128/16/164304/568362/Exploration-of-lattice-Hamiltonians-for-functional) ![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1\*FwG-cE5ABw\_KMUrkJ\_o2vQ.jpeg)
* [Active Learning for Deep Gaussian Process Surrogates](https://www.tandfonline.com/doi/full/10.1080/00401706.2021.2008505)
* [Actively learning GP dynamics](https://arxiv.org/abs/1911.09946)

#### General

* Review paper: https://arxiv.org/abs/2009.00236
* Book on Active learning: https://www.manning.com/books/human-in-the-loop-machine-learning
* "Towards Robust Deep Active Learning for Scientific Computing" https://arxiv.org/abs/2201.12632
* Deep bayesian active learning + image https://arxiv.org/abs/1703.02910
* A Comparative Survey of Deep Active Learning: https://arxiv.org/abs/2203.13450

### Tutorials/worksops

* [Google's Active Learning Playground](https://github.com/google/active-learning): This is a python module for experimenting with different active learning algorithms.
* [deep-active-learning](https://github.com/ej0cl6/deep-active-learning): Python implementations of the following active learning algorithms
* [PyTorch Active Learning](https://github.com/rmunro/pytorch\_active\_learning): Library for common Active Learning methods
* [active-learning-workshop](https://github.com/Azure/active-learning-workshop):KDD 2018 Hands-on Tutorial: Active learning and transfer learning at scale with R and Python. [PDF](https://github.com/Azure/active-learning-workshop/blob/master/active\_learning\_workshop.pdf)

### Codebases

1. Modal

Built on scipy+sklearn https://github.com/modAL-python/modAL

2. gpax

Gaussian processes + active learning! Very new, uses JAX ❤️ + numpyro Problem: unstable/

https://github.com/ziatdinovmax/gpax/

3. Scikit-activeml

> https://www.preprints.org/manuscript/202103.0194/v1 https://pypi.org/project/scikit-activeml/

Also built on scipy+sklearn

3. Baal

A " Bayesian active learning library" Built on pytorch Seems like focus on images https://baal.readthedocs.io/en/latest/notebooks/compatibility/sklearn\_tutorial/ https://baal.readthedocs.io/en/latest/ https://arxiv.org/abs/2006.09916

4. DeepAL

> https://github.com/ej0cl6/deep-active-learning

5. adaptive

> Adaptive sampling technique for 'learning' functional representation of the data https://adaptive.readthedocs.io/en/latest/index.html ![](https://adaptive.readthedocs.io/en/latest/\_static/logo\_docs.webm) https://github.com/python-adaptive/adaptive/tree/v1.0.0

6. AliPy Agnostic of pytorch/sklearn/tflow

https://github.com/NUAA-AL/alipy

7. libact Features the [active learning by learning](http://www.csie.ntu.edu.tw/\~htlin/paper/doc/aaai15albl.pdf) meta-strategy that allows the machine to automatically learn the best strategy on the fly.

Works with sklearn

Hasnt been updated in 2 years https://github.com/ntucllab/libact

## May 12, 2023

**1. LnL surface smooth** Ilya helped me find a bug, Lnl surface now looks smooth! woohoo!

![surface](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/8148ed8c-8a4d-455b-9aa2-3146d14c4d2f)

**2. bootstrapped uncertainty**

* generate additional COMPAS datasets\* and compute LnL
* use different Lnl to estimate a mean + std (to use in the GP uncertainty)

![sigma0\_lnl](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/6bbc96e8-02dc-40b9-a618-2502e15862c1) ![muz\_lnl](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/6dd8b01d-7631-4175-8aac-d151ce3cfba6)

**3. some injections**

Percentile-percentile plot made using 50 sets of {aSF, dSF, mu0, sigma0} posteriors.

Posteriors were generated by running inference w/ nested sampling on 50 mock BBH catalogs with our sklearn-GP trained lnLikelihood(aSF, dSF, mu0, sigma0 | mock BBH catalog). The sklearn-GP was given 1600 training points (and a bootstrapped LnL uncertainty). I used very low sampler settings to get things running quickly (nlive=250,)

![pp\_plot](https://github.com/avivajpeyi/compas\_ml\_surrogate/assets/15642823/7e72b7cb-903b-45ba-bb44-297bab339d39)

***

### NExt steps

* relative unc in lnl -- record the number of merging binaries -- record total -- record number in mock datasets -- active learning -- latin hyper cube -- sequentially sample -- is deepGP better or sklearn better? -- some training plots using deepGP and sklearn gp

### Paper thoughts

\-- compare with Jeff's work -- does my active learning improve computational costs? -- does emulating LnL help rather than training ChirpM Z? Emulating one number rather than several hundred emulators

## April 21, 2023

### Plots of LnL Surface (muz, sigma0)

* Black dashed line -- true injection SF parameter value (used to generate detection matrix)
* Blue stars -- sampled events from the detection matrix

|             | 2D                                                                                                                   | 1D                                                                                                                  |
| ----------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Injection 1 | ![surface\_1](https://user-images.githubusercontent.com/15642823/233517604-60f0f568-00a5-42ac-81ce-99ffa4f7995e.png) | ![plot1d\_1](https://user-images.githubusercontent.com/15642823/233517594-a958f3aa-4e45-4b67-9a3a-981223d08611.png) |
| Injection 2 | ![surface\_2](https://user-images.githubusercontent.com/15642823/233517897-4fd480f5-212f-4f43-aa06-99850c169737.png) | ![plot1d\_2](https://user-images.githubusercontent.com/15642823/233517895-5cd561c4-56e9-4224-84d7-0206acf48bcb.png) |

Some plots generated by:

* drawing random samples with prior-range
* computing likelihood and interpolating between points

| Full parameter space                                                                                           | Zoom                                                                                                                 | More data points in zoomed area                                                                                         |
| -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| ![orig](https://user-images.githubusercontent.com/15642823/229909634-025e6918-acbc-4387-a402-021fa26ba1e5.png) | ![orig\_zoom](https://user-images.githubusercontent.com/15642823/229909632-92fae747-6cfd-47a8-998e-f10d6f807577.png) | ![like\_focused](https://user-images.githubusercontent.com/15642823/229909618-0d996ab1-ab6d-425a-a7bd-6a35d110956b.png) |

A different version of plot:

* linearly spaced **grid** of points
* top: scatter plot with likelihood values colored
* mid: interpolate between scatter plot points
* bot: `imshow( log_likelihood - max(log_likelihood))` of data (no interpolation)
* limit color bar from 0 to -1

| ln likelihood                                                                                                             | likelihood                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| ![lnl\_focused\_0](https://user-images.githubusercontent.com/15642823/229909967-836f454b-acc8-4872-b630-4f4f4b391d3d.png) | ![like\_focused\_1](https://user-images.githubusercontent.com/15642823/229909976-d7b4a581-e097-4610-9d05-efe76a35feff.png) |

**Zooming in further** I am computing the likelihood and zooming in further -- this will take some time

### BOOTSTRAP LnL uncertainty

* resample population with N + sqrt(N) (with replacements)
* compute list of LnL with different populations and obtain a boot-strapped LnL

### Talk with Indranil

* LnL surface quite unsmooth, but just near the peak

1. Maybe try to figure about a hack to "smooth out" the likelihood
2. Need to ensure that training data is generated near the peak -- maybe something like 'subset simulation' to hone in on 'rare' events

https://www.researchgate.net/publication/320798357\_A\_short\_report\_on\_PYTHON\_implementation\_of\_subset\_simulation https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.24900 https://github.com/jpmit/lennyffs https://github.com/hackl/pyre https://archive.org/details/arxiv-1104.3667

## March 11, 2023

### Exploring the number of training points needed

Tried sampler after training with \[10 - 10 000] data points (spaced by 100). The sampler failed (nburn > nsteps, with nsteps=1000).

| n    | image                                                                                                                             |
| ---- | --------------------------------------------------------------------------------------------------------------------------------- |
| 100  | ![surr\_run\_100\_corner](https://user-images.githubusercontent.com/15642823/221540450-756eb4b1-02c3-482c-8490-24d37a86ef25.png)  |
| 500  | ![surr\_run\_500\_corner](https://user-images.githubusercontent.com/15642823/221540477-c38d2bc2-2e22-41cc-ac00-98ddfb3132df.png)  |
| 1000 | ![surr\_run\_1000\_corner](https://user-images.githubusercontent.com/15642823/221540485-e9bc1544-a1f8-4f71-9234-d0b8e7c39c8e.png) |

In this case, the surrogate has trained with \~500 training points.

### PP Test

Running a PP-test with 100 events (500 training points for each). Still running, but atm

1. Sampler fails for several injection sets
2. Sometimes, the sampler finishes and the posterior looks bogus (maybe we need to increase training data/adaptive sampling).

![pp\_plot\_1](https://user-images.githubusercontent.com/15642823/227379476-35d36827-587b-49ab-b988-3b23d90151c4.png) This type of pp-plot means we have over-constrained the posterior

Examples:

| idx  | detection-matrix                                                                                                    | posterior                                                                                                                 | comments                      |
| ---- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| 628  | ![uni\_628](https://user-images.githubusercontent.com/15642823/221544748-24d03710-0fb3-4311-a3c0-0a3cfd63a291.png)  | ![posterior\_628](https://user-images.githubusercontent.com/15642823/221544732-3bb4f112-d4dd-47d1-a41e-81de206d18b5.png)  | Looks ok..                    |
| 1071 | ![uni\_1071](https://user-images.githubusercontent.com/15642823/221545027-a4a3cdcf-6284-4552-8cc8-94da74be20cc.png) | ![posterior\_1071](https://user-images.githubusercontent.com/15642823/221545017-b12b90f8-f571-462a-abf6-781c4d3e9824.png) | Sampler didn't converge       |
| 1892 | ![uni\_1892](https://user-images.githubusercontent.com/15642823/221545900-b608d09c-8d85-4632-ba9a-2dbde6d7f109.png) |                                                                                                                           | Sampler failed (nburn>nsteps) |

### Investigating a 'biased' posterior

Simulating the following system:

| Data                                                                                                               | Corner                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| ![uni\_tru](https://user-images.githubusercontent.com/15642823/222046958-28e9b094-49bb-472d-8787-86e464b4be51.png) | ![corner\_lnl\_true](https://user-images.githubusercontent.com/15642823/222046961-d9a835f9-f446-4bce-b03f-4b25bbddccd9.png) |

THIS IS THE TRUE EVENT -- True Lnl: -295.35

Now, trying to train with different amounts of data: ![learning\_curve](https://user-images.githubusercontent.com/15642823/222046956-86237ac7-af06-42a3-9b96-8f6acd980936.png)

The validation error is really high!!

Here is a plot of the numerical Lnl vs surrogate Lnl when the surrogate is trained with \~400 datapoints: ![model\_diagnostic\_err](https://user-images.githubusercontent.com/15642823/227391978-21b8c6ba-c6b9-4561-92ee-0761608dfac2.png)

**QS FOR ILYA** What is the numerical LnL uncertainty? I should try to account for that in the GP (right now I am assuming our numerical LnL measurements are perfect). Unc in COMPAS? Unc in comsic integrator?

Sampling with different trained models (each using more data):

| Ntrain | Corner                                                                                                                  | true - pred LnL |
| ------ | ----------------------------------------------------------------------------------------------------------------------- | --------------- |
| 100    | ![n100\_corner](https://user-images.githubusercontent.com/15642823/222046954-4edd4719-39db-440f-9fa7-7a544a8884c4.png)  | 0               |
| 250    | ![n250\_corner](https://user-images.githubusercontent.com/15642823/222046948-604cd760-830f-4ca7-b6cf-eeb3d4f15cc0.png)  | 0               |
| 500    | ![n500\_corner](https://user-images.githubusercontent.com/15642823/222046943-64248ea1-c726-4977-b09a-123629aade60.png)  | 0.19            |
| 1000   | ![n1000\_corner](https://user-images.githubusercontent.com/15642823/222046929-b299a774-81f6-4265-9751-7d6ffea3c089.png) | 0.23            |

***

#### inf Lnl bug at high muz values!

Some regions of the parameter space --> lnl \~ -inf

| Uni                                                                                                                     | Lnl Training Data                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| ![uni\_inf\_lnl](https://user-images.githubusercontent.com/15642823/227390997-132ce3d3-ec24-41eb-b3de-1dc6b552e621.png) | ![cache\_inf\_lnl](https://user-images.githubusercontent.com/15642823/227390875-05a644c3-ac2b-4ee5-bae9-097a022ddf45.png) |

***

### Investigating model with different numbers of training points

![uni\_inv](https://user-images.githubusercontent.com/15642823/227399574-3945c7da-564d-46da-8a0f-de830ebcc0e8.png) ![cache\_inv](https://user-images.githubusercontent.com/15642823/227399595-27c1ca16-664c-41cb-b6e9-26644161179a.png)

| N    | pic                                                                                                                   |
| ---- | --------------------------------------------------------------------------------------------------------------------- |
| 100  | ![model\_100](https://user-images.githubusercontent.com/15642823/227399718-c7d7a498-4fee-4615-aad7-dc4d96155353.png)  |
| 500  | ![model\_500](https://user-images.githubusercontent.com/15642823/227399707-82d6128f-e9a3-45c3-8ce6-77c8aaf66b31.png)  |
| 1000 | ![model\_1000](https://user-images.githubusercontent.com/15642823/227399803-4ae3e6e6-7e2b-4c44-8b0d-eecfef2c755e.png) |
| 2000 | ![model\_2000](https://user-images.githubusercontent.com/15642823/227399675-d25a2201-80d5-4b00-92af-ab959585e5ec.png) |

![learning\_curve](https://user-images.githubusercontent.com/15642823/227402638-77d8aa6a-0faf-44d6-8d4a-7b545b86f400.png)
