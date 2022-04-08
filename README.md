# Transfer learning for Bayesian optimization
The codes are written based on [this repository](https://github.com/automl/transfer-hpo-framework).

In this codebase, we are trying to reproduce the results of TST-R and RGPE using the TAF acquisition function in [`Practical transfer learning for Bayesian optimization`](https://arxiv.org/pdf/1802.02219v3.pdf).

You can find the information for RGPE in the paper and find the information that for TST-R in the paper [`Two-stage transfer surrogate model for automatic hyperparameter optimization`](https://www.ismll.uni-hildesheim.de/pub/pdfs/wistuba_et_al_ECML_2016.pdf).

## Initial setup

```shell
$ conda install swig
$ pip install -r requirements.txt
```
