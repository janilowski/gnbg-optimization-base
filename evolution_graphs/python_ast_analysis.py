"""
This module analyzes Python source code using Abstract Syntax Trees (AST) and graphs 
to extract code characteristics and graph metrics. It includes functionality to 
calculate code complexity, analyze graph properties, and visualize the AST as a graph.

Classes:
    - BuildAST: Traverses an AST to construct a directed graph representing the structure.

Functions:
    - code_compare(code1, code2, printdiff=False): Compares two code snippets for similarity based on their differences.
    - analyse_complexity(code): Calculates code complexity metrics such as cyclomatic complexity and token count.
    - eigenvector_centrality_numpy(G, max_iter=500): Computes the eigenvector centrality for a graph.
    - analyze_graph(G): Extracts and computes various graph metrics including depth, degree, and clustering properties.
    - visualize_graph(G): Visualizes a graph and saves it as a PDF.
    - process_file(path, visualize): Reads a Python file, builds an AST graph, computes stats, and optionally visualizes the graph.
    - process_code(python_code, visualize): Processes a Python code string, builds an AST graph, computes stats, and optionally visualizes the graph.
    - aggregate_stats(results): Aggregates statistics from multiple graph analysis results.

Dependencies:
    - argparse, ast, difflib, json, jellyfish, jsonlines, lizard, matplotlib, networkx, numpy, scipy, tqdm
"""

import argparse
import ast
import difflib
import json
import math
import re
import warnings

import lizard
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tqdm
from scipy.stats import entropy


GRAPH_STAT_KEYS = [
    "Nodes",
    "Edges",
    "Max Degree",
    "Min Degree",
    "Mean Degree",
    "Degree Variance",
    "Transitivity",
    "Max Depth",
    "Min Depth",
    "Mean Depth",
    "Max Clustering",
    "Min Clustering",
    "Mean Clustering",
    "Clustering Variance",
    "Degree Entropy",
    "Depth Entropy",
    "Assortativity",
    "Average Eccentricity",
    "Diameter",
    "Radius",
    "Edge Density",
    "Average Shortest Path",
]

COMPLEXITY_STAT_KEYS = [
    "mean_complexity",
    "total_complexity",
    "mean_token_count",
    "total_token_count",
    "mean_parameter_count",
    "total_parameter_count",
]


def empty_stats(keys):
    return {key: np.nan for key in keys}


def normalize_metric_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return np.nan
    return value


def normalize_stats(stats):
    return {key: normalize_metric_value(value) for key, value in stats.items()}


def format_error(exc):
    if isinstance(exc, SyntaxError):
        location = f" line {exc.lineno}" if exc.lineno is not None else ""
        return f"{type(exc).__name__}{location}: {exc.msg}"
    return f"{type(exc).__name__}: {exc}"


def extract_analysis_note(code):
    fields = {}
    in_note = False
    for raw_line in code.splitlines():
        line = raw_line.strip()
        if line == "# ALGORITHM_ANALYSIS_NOTE_BEGIN":
            in_note = True
            continue
        if line == "# ALGORITHM_ANALYSIS_NOTE_END":
            break
        if not in_note or not line.startswith("#"):
            continue

        content = line[1:].strip()
        if ":" not in content:
            continue
        key, value = content.split(":", 1)
        normalized_key = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
        if normalized_key:
            fields[f"note_{normalized_key}"] = value.strip()
    return fields


def code_compare(code1, code2, printdiff=False):
    # Parse the Python code into ASTs
    # Use difflib to find differences
    diff = difflib.ndiff(code1.splitlines(), code2.splitlines())
    # Count the number of differing lines
    diffs = sum(1 for x in diff if x.startswith("- ") or x.startswith("+ "))
    # Calculate total lines for the ratio
    total_lines = max(len(code1.splitlines()), len(code2.splitlines()))
    similarity_ratio = (total_lines - diffs) / total_lines if total_lines else 1
    return 1 - similarity_ratio


def analyse_complexity(code):
    # Analyse the code complexity of the code.
    try:
        i = lizard.analyze_file.analyze_source_code("algorithm.py", code)
    except Exception:
        return empty_stats(COMPLEXITY_STAT_KEYS)

    complexities = []
    token_counts = []
    parameter_counts = []
    for f in i.function_list:
        complexities.append(getattr(f, "cyclomatic_complexity", 0))
        token_counts.append(getattr(f, "token_count", 0))
        parameter_counts.append(len(getattr(f, "full_parameters", [])))

    if not complexities:
        return {key: 0.0 for key in COMPLEXITY_STAT_KEYS}

    return normalize_stats(
        {
            "mean_complexity": np.mean(complexities),
            "total_complexity": np.sum(complexities),
            "mean_token_count": np.mean(token_counts),
            "total_token_count": np.sum(token_counts),
            "mean_parameter_count": np.mean(parameter_counts),
            "total_parameter_count": np.sum(parameter_counts),
        }
    )


# Parse Python AST and build a graph
class BuildAST(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_node = 0
        self.node_stack = []

    def generic_visit(self, node):
        node_id = self.current_node
        self.graph.add_node(node_id, label=type(node).__name__)

        if self.node_stack:
            parent_id = self.node_stack[-1]
            self.graph.add_edge(parent_id, node_id)

        self.node_stack.append(node_id)
        self.current_node += 1

        super().generic_visit(node)

        self.node_stack.pop()

    def build_graph(self, root):
        self.visit(root)
        return self.graph


def eigenvector_centrality_numpy(G, max_iter=500):
    try:
        return (nx.eigenvector_centrality_numpy(G, max_iter=500),)
    except Exception:
        return np.nan


# Function to extract graph characteristics
def analyze_graph(G):
    if G.number_of_nodes() == 0:
        stats = empty_stats(GRAPH_STAT_KEYS)
        stats["Nodes"] = 0
        stats["Edges"] = 0
        return stats

    depths = dict(nx.single_source_shortest_path_length(G, min(G.nodes())))
    degrees = sorted((d for n, d in G.degree()), reverse=True)
    leaf_depths = [
        depth for node, depth in depths.items() if G.out_degree(node) == 0
    ]  # depth from root to leaves
    if not leaf_depths:
        leaf_depths = list(depths.values())
    clustering_coefficients = list(nx.clustering(G).values())
    # Additional Features (not in paper)
    # Convert the directed graph to an undirected graph to avoid SCC problems
    undirected_G = G.to_undirected()
    if G.number_of_nodes() == 1:
        diameter = 0
        radius = 0
        avg_shortest_path = 0
        avg_eccentricity = 0
    elif nx.is_connected(undirected_G):  # check if undirected graph is connected
        diameter = nx.diameter(undirected_G)
        radius = nx.radius(undirected_G)
        avg_shortest_path = nx.average_shortest_path_length(undirected_G)
        avg_eccentricity = np.mean(list(nx.eccentricity(undirected_G).values()))
    else:
        # Calculate path metrics on the largest weakly connected component.
        largest_cc = max(nx.connected_components(undirected_G), key=len)
        subgraph = undirected_G.subgraph(largest_cc).copy()
        diameter = nx.diameter(subgraph)
        radius = nx.radius(subgraph)
        avg_shortest_path = nx.average_shortest_path_length(subgraph)
        avg_eccentricity = np.mean(list(nx.eccentricity(subgraph).values()))
    edge_density = (
        G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1))
        if G.number_of_nodes() > 1
        else 0
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assortativity = nx.degree_assortativity_coefficient(G)

    return normalize_stats(
        {
        # Number of Nodes and Edges
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "Max Degree": max(degrees),
            "Min Degree": min(degrees),
            "Mean Degree": np.mean(degrees),
            "Degree Variance": np.var(degrees),
        # Transitivity
            "Transitivity": nx.transitivity(G),
        # Depth analysis
            "Max Depth": max(leaf_depths),
            "Min Depth": min(leaf_depths),
            "Mean Depth": np.mean(leaf_depths),
            "Max Clustering": max(clustering_coefficients),
            "Min Clustering": min(clustering_coefficients),
            "Mean Clustering": nx.average_clustering(G),
            "Clustering Variance": np.var(clustering_coefficients),
        # Entropy
            "Degree Entropy": entropy(degrees),
            "Depth Entropy": entropy(leaf_depths),
            "Assortativity": assortativity,
            "Average Eccentricity": avg_eccentricity,
            "Diameter": diameter,
            "Radius": radius,
            "Edge Density": edge_density,
            "Average Shortest Path": avg_shortest_path,
        }
    )


def visualize_graph(G):
    try:
        from networkx.drawing.nx_pydot import graphviz_layout

        pos = graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    labels = nx.get_node_attributes(G, "label")
    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        font_size=8,
        font_weight="bold",
        arrows=True,
    )
    plt.savefig("graph1.pdf")


# Function to create graph out of AST
def process_file(path, visualize=False):
    with open(path, "r") as file:
        python_code = file.read()
    return process_code(python_code, visualize=visualize)


# Function to create graph out of AST
def process_code(python_code, visualize=False):
    result = {
        "parse_ok": True,
        "graph_ok": True,
        "complexity_ok": True,
        "error": "",
    }
    graph = None

    try:
        root = ast.parse(python_code)
        build = BuildAST()
        graph = build.build_graph(root)
        stats = analyze_graph(graph)
    except Exception as exc:
        stats = empty_stats(GRAPH_STAT_KEYS)
        result["parse_ok"] = not isinstance(exc, SyntaxError)
        result["graph_ok"] = False
        result["error"] = format_error(exc)

    if visualize and graph is not None:
        try:
            visualize_graph(graph)
        except Exception as exc:
            result["graph_ok"] = False
            result["error"] = "; ".join(
                part for part in [result["error"], format_error(exc)] if part
            )

    complexity_stats = analyse_complexity(python_code)
    if any(isinstance(value, float) and np.isnan(value) for value in complexity_stats.values()):
        result["complexity_ok"] = False

    return {
        **result,
        **normalize_stats(stats),
        **complexity_stats,
        **extract_analysis_note(python_code),
    }


def aggregate_stats(results):
    print("Aggregate Statistics:")
    print("Total Nodes:", sum(result["Nodes"] for result in results))
    print("Total Edges:", sum(result["Edges"] for result in results))
    print(
        "Average Transitivity:",
        sum(result["Transitivity"] for result in results) / len(results),
    )
    print("Max Depth:", max(result["Max Depth"] for result in results))
    print(
        "Average Degree Mean:", np.mean([result["Mean Degree"] for result in results])
    )
    print(
        "Average Clustering Coefficient:",
        np.mean([result["Mean Clustering"] for result in results]),
    )
    print(
        "Average Eccentricity:",
        np.mean([result["Average Eccentricity"] for result in results]),
    )
    print(
        "Average Edge Density:", np.mean([result["Edge Density"] for result in results])
    )

