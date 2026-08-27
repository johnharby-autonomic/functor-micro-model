Probability Functor Micro Model


A compact reference implementation of a Probability Functor Model used for Functor Model benchmarking and experimentation.


This repository contains the micro-model implementation used to evaluate the computational characteristics of function-based learning and inference. It is intended as a reproducible research artifact rather than a production machine-learning framework. Contact jharby@functormodel.ai if you have interests in the production version.


Usage


1.)
python3 -m core.prob_functor_model —mode lifecycle
This will run a quick iteration with 100 model updates and 1000 predicitons


2.)
From the root folder, run ./run_sweep.sh.
You can edit this script to select routes, lazy/eager moment evaluation, and
the number of updates and predictions per update. The console output gives a
CSV report (route,strategy,commit_us,predict_us,lifecycle_us_per_update,...),
and sweep_results.json records the measurements, individual trial timings,
state hashes, and runtime metadata.


Overview


Conventional neural models primarily encode learned behavior through parameter optimization:

theta_old -> theta_new

The Functor Model approach instead treats the learned function and its validated transformations as first-class objects:

f_new = the direct sum of f_old and delta

The Probability Functor Model extends this idea to probabilistic behavior. An input may map either to a determined result or to a probability measure when uncertainty remains.

Conceptually,

F(x) = { delta{f(x)} for x in D
         u(x) for x in U. }

where:

(D) is the determined region,

(U) is the uncertain region,

delta_{f(x)} is a Dirac measure concentrated at the determined result,

u(x) represents unresolved probabilistic behavior.


This allows probabilistic computation to be concentrated where uncertainty exists rather than necessarily applying the same probabilistic processing to every input.


​
