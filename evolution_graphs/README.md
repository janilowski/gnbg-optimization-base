## Paper that was the basis for the implementation

title: Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery
date: 4 Jul 2025
authors: Niki van Stein, Haoran Yin, Anna V. Kononova, Thomas Bäck, Gabriela Ochoa
DOI: https://doi.org/10.48550/arXiv.2507.03605

## What is included

The AST/static-code feature extraction as described in the paper’s Code Evolution Graph concept. The implementation extracts: graph node/edge counts, degree statistics, clustering, depth, entropy, assortativity, path metrics, and complexity/token/parameter metrics for every candidate.

The batch extractor groups candidates by model folder, assigns a stable pseudo-order alg_id, writes one CSV row per candidate, and records parse/graph/complexity failures.

The BBO visualization script is focused on BBO AST graph statistics and avoids the old mixed BP/TSP assumptions. It produces PCA, t-SNE, and optional UMAP projections from AST features, plus feature and complexity evolution plots.

## What is not included

### Algos behavioral analysis based on full bbo computations using candidate algorithms

The visualization is not a true Code Evolution Graph in the paper’s sense. The paper’s CEG has nodes as generated algorithms, directed parent→offspring edges, token count over generations, node sizing by parent-selection frequency, and performance coloring.

The implementation instead uses filename-derived alg_id as a pseudo-evolution index, explicitly leaves parent_id / parent_ids empty, and sets `lineage_available=False`.

The paper’s main methodology depends on complete optimization traces and computes behaviour metrics for exploration/diversity, exploitation/intensification, convergence progress, and stagnation/reliability. I did not see implementations for those trace-based BBO metrics in the uploaded Python files.

The paper also constructs behaviour-based Search Trajectory Networks using five least-correlated behaviour metrics — Expl. %, Conv-rate, Δ fitness, Success %, and No-imp streak — then partitions behaviour space into hypercubes and counts node/edge sampling frequencies. The uploaded implementation does not build STNs.

The paper uses parallel-coordinate plots over behaviour metrics, with runs colored by performance quartile. The uploaded implementation does not create parallel-coordinate plots over behaviour metrics.
