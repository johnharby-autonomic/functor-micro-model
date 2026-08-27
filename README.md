# Probability Functor Micro Model

A compact reference implementation of a **Probability Functor Model** used for Functor Model benchmarking and experimentation.

This repository contains the micro-model implementation used to evaluate the computational characteristics of function-based learning and inference. It is intended as a reproducible research artifact rather than a production machine-learning framework. Contact jharby@functormodel.ai if you have interests in the production version.

## Usage

1.) 
python3 -m core.prob_functor_model --mode lifecycle 
This will run a quick iteration with 100 model updates and 1000 predicitons

2.) 
From the root folder, run `./run_sweep.sh`.
You can edit this script to select routes, lazy/eager moment evaluation, and
the number of updates and predictions per update. The console output gives a
CSV report (`route,strategy,commit_us,predict_us,lifecycle_us_per_update,...`),
and `sweep_results.json` records the measurements, individual trial timings,
state hashes, and runtime metadata.

## Overview

Conventional neural models primarily encode learned behavior through parameter optimization:

\[
\theta_t \longrightarrow \theta_{t+1}.
\]

The Functor Model approach instead treats the learned function and its validated transformations as first-class objects:

\[
f_t \xrightarrow{\Delta_t} f_{t+1}.
\]

The Probability Functor Model extends this idea to probabilistic behavior. An input may map either to a determined result or to a probability measure when uncertainty remains.

Conceptually,

\[
F(x)=
\begin{cases}
\delta_{f(x)}, & x\in D,\\
\mu_x, & x\in U,
\end{cases}
\]

where:

- \(D\) is the determined region,
- \(U\) is the uncertain region,
- \(\delta_{f(x)}\) is a Dirac measure concentrated at the determined result,
- \(\mu_x\) represents unresolved probabilistic behavior.

This allows probabilistic computation to be concentrated where uncertainty exists rather than necessarily applying the same probabilistic processing to every input.

## Why "Micro"?

This implementation is deliberately small.

The purpose is to isolate and measure the core execution mechanism without the infrastructure and computational overhead of a large model. It provides an experimental environment for studying:

- function-based learning,
- low-cost inference,
- determined versus uncertain execution,
- local model updates,
- one-shot updates,
- model evolution,
- latency and throughput,
- CPU execution,
- energy consumption.

It should not be interpreted as a claim that a micro model is a replacement for every neural-network or foundation-model workload.

## Learning Model

A learning event produces a change to executable model behavior:

\[
f_t \xrightarrow{\Delta_t} f_{t+1}.
\]

The resulting evolution can be represented as a sequence:

\[
f_0 \xrightarrow{\Delta_1} f_1
\xrightarrow{\Delta_2} \cdots
\xrightarrow{\Delta_n} f_n.
\]

This provides an explicit representation of model evolution rather than treating learning solely as an opaque change to a parameter vector.

For probabilistic regions, an admissible local update may take the form

\[
p'(y)=p(y)+\Delta(y),
\]

with \(\Delta\) supported only on an uncertain region \(U\).

For a probability density, admissibility requires at minimum:

\[
p+\Delta\geq 0
\]

and, for a strictly local mass-preserving update,

\[
\int_U \Delta(y)\,dy=0.
\]

This permits probability mass to be redistributed locally without modifying the model outside the affected region.

## Benchmarking

This repository contains the implementation used in our Probability Functor Micro benchmarks.

Benchmark results should be reported together with sufficient information to reproduce them, including:

- hardware and processor architecture,
- operating system,
- runtime and compiler versions,
- dataset and workload,
- number of training and inference operations,
- warm-up methodology,
- elapsed time,
- throughput,
- latency distribution,
- CPU utilization,
- memory consumption,
- energy-measurement methodology.

Where energy measurements are reported, they should be treated as measurements of the specified hardware and workload rather than universal energy requirements of the architecture.

## Research Questions

The implementation is intended to support experimental investigation of questions including:

1. When can learned behavior be represented efficiently as functional updates rather than repeated parameter optimization?
2. How much computation can be avoided when determined inputs follow a deterministic execution path?
3. Can probabilistic learning be localized to uncertain regions of the model?
4. When can previously learned functional deltas be reused?
5. How do function-based updates compare with conventional retraining in latency, energy consumption, accuracy, and retention?
6. Can sequential functional updates provide an effective mechanism for continual learning?
7. How does the approach behave under distribution and concept drift?

These are research questions. Results from individual benchmarks should not be generalized beyond the conditions under which they were measured.

## Relationship to Functor Model Architecture

Probability Functor Micro is a small experimental implementation within the broader **Functor Model Architecture (FcMA)** research program.

FcMA investigates models in which functions, compositions, transformations, topology, provenance, and governed model evolution are treated explicitly.

The micro implementation intentionally contains only the mechanisms required for the experiments represented in this repository. It should therefore not be interpreted as a complete implementation of FcMA.

## Reproducibility

Reproduction and independent evaluation are encouraged.

When comparing this implementation with
