# By Amy Becker

import random
a = random.randint(0,10000000000)
import networkx as nx
from gerrychain.random import random
random.seed(a)
import csv
import os
from functools import partial
import json
import random
import numpy as np
import sys

import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import time

from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept,
    constraints,
    updaters,
)
from gerrychain.proposals import recom
from gerrychain.updaters import cut_edges, Tally
from gerrychain.tree import PopulatedGraph, recursive_tree_part, predecessors, bipartition_tree, random_spanning_tree, find_balanced_edge_cuts_memoization
from networkx.algorithms import tree
from collections import deque, namedtuple


def gdf_print_map(partition, filename, gdf, unit_name):
    cdict = {partition.graph.nodes[i][unit_name]:partition.assignment[i] for i in partition.graph.nodes()}
    gdf['color'] = gdf.apply(lambda x: cdict[x[unit_name]], axis=1)
    plt.figure()
    gdf.plot(column='color', cmap = 'tab10')
    plt.savefig(filename+'.png', dpi = 600)
    plt.close("all")


def population_deviation(partition):
    ideal_pop = sum(partition["population"].values())/len(partition)
    max_over = (max(partition["population"].values())-ideal_pop)/ideal_pop
    min_under = (min(partition["population"].values())-ideal_pop)/ideal_pop
    return max(abs(max_over),abs(min_under))



def bipartition_tree_alt(
    graph,
    pop_col,
    pop_target,
    epsilon,
    node_repeats=1,
    spanning_tree=None,
    spanning_tree_fn=random_spanning_tree,
    balance_edge_fn=find_balanced_edge_cuts_memoization,
    choice=random.choice,
    max_attempts = float('inf')
):
    # basically identical to bipartiion tree, just adds a 'stop after x tries' condition to avoid stalling and puts main function in a try to avoid failing at balance errors
    populations = {node: graph.nodes[node][pop_col] for node in graph}
    possible_cuts = []
    if spanning_tree is None:
        spanning_tree = spanning_tree_fn(graph)
    restarts = 0
    attempts = 0
    while len(possible_cuts) == 0 and attempts < max_attempts:
        if restarts == node_repeats:
            spanning_tree = spanning_tree_fn(graph)
            restarts = 0
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        try:
            possible_cuts = balance_edge_fn(h, choice=choice)
        except:
            possible_cuts = []
        restarts += 1
        attempts += 1
    if len(possible_cuts) == 0:
        return None
    return choice(possible_cuts).subset


def pop_shuffle(partition, pop_col, epsilon, node_repeats = 0, method =bipartition_tree, weights = False):
    #determine adjacent districts and which cut edges are between those pairs
    adj_dists = set([(min(partition.assignment[i],partition.assignment[j]),max(partition.assignment[i],partition.assignment[j])) for (i,j) in partition["cut_edges"]])
    cut_edge_dict = {i:[] for i in adj_dists}
    for (i,j) in partition["cut_edges"]:
        cut_edge_dict[(min(partition.assignment[i],partition.assignment[j]),max(partition.assignment[i],partition.assignment[j]))].append((i,j))
    #choose a neighboring pair either at random or proportional to the population deviation of the pair (may want to play around with other weighting schemes for how to choose next pair)
    #should have SOME variation in which pair to pick, otherwise can get stuck in loop picking same pair each time
    neighbor_pair = random.choices(tuple(adj_dists), weights = [abs(partition.population[pair[0]] - partition.population[pair[1]]) for pair in adj_dists] if weights else [1 for pair in adj_dists])[0]
    edge = random.choice(cut_edge_dict[neighbor_pair])
    parts_to_merge = (partition.assignment[edge[0]], partition.assignment[edge[1]])
    subgraph = partition.graph.subgraph(partition.parts[parts_to_merge[0]] | partition.parts[parts_to_merge[1]])
    pop_target = (partition.population[neighbor_pair[0]] + partition.population[neighbor_pair[1]])/2
    
    nodes = bipartition_tree_alt(
            subgraph,
            pop_col=pop_col,
            pop_target=pop_target,
            epsilon=epsilon,
            node_repeats=node_repeats,
            max_attempts = 50   #may want to play with adjusting this number
        )
    if nodes is None:   #avoids stalling
        return partition
    flips = {}
    for node in nodes:
        flips[node] = parts_to_merge[0]
    for node in set(subgraph.nodes) - nodes:
        flips[node] = parts_to_merge[-1]
    return partition.flip(flips)
    
def gen_initial_partition(
    graph,
    num_districts,
    my_updaters,
):
    random_assign = {v:0 for v in graph.nodes()}
    #randomly choose single-unit district seeds.  May be smarter ways to do this! or may want to start with larger district seeds like designating an entire county or municipality to be district i, rather than just a single unit
    random_nodes = [0]+random.sample([v for v in graph.nodes() if v!= 0],num_districts-1)
    random_assign.update({random_nodes[i]:i for i in range(num_districts)})
    initial_partition = Partition(graph = graph, assignment = random_assign, updaters = my_updaters)
    return initial_partition
