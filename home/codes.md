# 🤖 Codes

The logic has been split into 3 main repos:

|                                                                         |                                                                              |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 🖥️ [lnl\_computer](https://github.com/COMPAS-Surrogate/lnl\_computer)  | Generates detection matrices + caches LnL values (incl slurm utils)          |
| 🧬 [lnl\_surrogate](https://github.com/COMPAS-Surrogate/lnl\_surrogate) | Builds a LnL surrogate (and decides next points needed to improve surrogate) |
| ⚙️ [pipeline](https://github.com/COMPAS-Surrogate/pipeline)             | Runs the computer and surrogate builder                                      |

There are some more non-essential repos for the project

* [paper](https://github.com/COMPAS-Surrogate/paper)
* [active learning experiments](https://github.com/COMPAS-Surrogate/active\_learning\_experiments)
* [surrogate\_studies](https://github.com/COMPAS-Surrogate/surrogate\_studies)
