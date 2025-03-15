import geopandas as gpd
import networkx as nx
import random
from shapely.strtree import STRtree
from rtree import index
from shapely.geometry import Polygon
from typing import Dict, Set


def build_graph_from_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    graph = nx.Graph()

    print("The # of nodes in the graph is: ", len(gdf))

    # Add nodes (polygons)
    for idx, row in gdf.iterrows():
        graph.add_node(idx)

    # Create a spatial index
    spatial_index = STRtree(gdf.geometry)

    # Add edges based on shared boundaries
    for idx, row in gdf.iterrows():
        possible_matches_index = spatial_index.query(row.geometry)
        possible_matches = gdf.iloc[possible_matches_index]
        for idx2, row2 in possible_matches.iterrows():
            if idx != idx2 and row.geometry.intersects(row2.geometry):
                graph.add_edge(idx, idx2)

    return graph, gdf


def build_graph_from_metis(metis_file_path):
    graph = nx.read_adjlist(metis_file_path)
    print("The # of nodes in the graph is: ", graph.number_of_nodes())
    return graph


def checkFeasible(G, regions, unassigned):
    return not unassigned


def build_parent_map(graph, gdf):
    parent_map = {}
    # if gdf is None:
    #     return parent_map
    #
    # # Step 1: Build R-tree spatial index and track node positions
    # idx = index.Index()
    # node_id_to_pos = {}
    # for pos, node in enumerate(graph.nodes()):
    #     if node not in gdf.index:
    #         raise ValueError(f"Node {node} is missing in the GeoDataFrame.")
    #     geom = gdf.loc[node, 'geometry']
    #     idx.insert(pos, geom.bounds)
    #     node_id_to_pos[node] = pos
    #
    # # Step 2: Iterate over all nodes to find parent-child relationships
    # for parent in graph.nodes():
    #     parent_geom = gdf.loc[parent, 'geometry']
    #     p_minx, p_miny, p_maxx, p_maxy = parent_geom.bounds
    #
    #     # Query R-tree for candidate positions
    #     candidates = list(idx.intersection((p_minx, p_miny, p_maxx, p_maxy)))
    #     candidate_nodes = [
    #         node for node in graph.nodes()
    #         if node_id_to_pos[node] in candidates
    #     ]
    #
    #     children = set()
    #     for child in candidate_nodes:
    #         if child == parent:
    #             continue  # Skip self
    #
    #         child_geom = gdf.loc[child, 'geometry']
    #         if parent_geom.contains(child_geom):
    #             children.add(child)
    #
    #     # Optional: Recursively add grandchildren
    #     stack = list(children)
    #     while stack:
    #         current_child = stack.pop()
    #         grand_children = parent_map.get(current_child, set())
    #         children.update(grand_children)
    #         stack.extend(grand_children)
    #
    #     if children:
    #         parent_map[parent] = children

    return parent_map


def region_growing(
        G,
        target_cardinality,
        unassigned,
        seeds,
        max_retries: int = 10):
    for _ in range(max_retries):
        # Select seed (prioritize neighbors of existing regions)
        if seeds:
            seed = random.choice(seeds)
            if seed not in unassigned:
                seeds.remove(seed)
                continue
        else:
            if not unassigned:
                return None
            seed = random.choice(list(unassigned))

        region = {seed}
        frontier = set(G.neighbors(seed)) & unassigned

        while len(region) < target_cardinality and frontier:
            next_area = random.choice(list(frontier))
            region.add(next_area)
            unassigned.remove(next_area)
            frontier = (frontier - {next_area}
                        ) | (set(G.neighbors(next_area)) & unassigned)

        if len(region) == target_cardinality:
            return region
        else:
            # Rollback and retry
            unassigned.update(region)
            region.clear()

    return None


def region_merging(
        G,
        region,
        unassigned):
    components = list(nx.connected_components(G.subgraph(unassigned)))
    if len(components) <= 1:
        return region, unassigned

    largest_component = max(components, key=len)
    for component in components:
        if component != largest_component:
            region.update(component)
            unassigned.difference_update(component)

    return region, unassigned


def region_splitting(
    G,
    parent_map,
    merged_region,
    target_cardinality,
    unassigned,
    allowed_cardinalities,
    max_attempts=100
):
    valid_region = set(merged_region)
    excess_areas = set()
    removed_set = set()  # Track areas removed during shrinking
    attempts = 0

    # Shrink phase
    while len(valid_region) > target_cardinality and attempts < max_attempts:
        boundary = [n for n in valid_region if any(
            nb in unassigned for nb in G.neighbors(n))]
        if not boundary:
            break
        node_to_remove = random.choice(boundary)
        nested_children = parent_map.get(node_to_remove, set())
        to_remove = {node_to_remove} | (nested_children & valid_region)
        valid_region.difference_update(to_remove)
        removed_set.update(to_remove)  # Track removed areas
        unassigned.update(to_remove)

        # Ensure contiguity after removal
        comps = list(nx.connected_components(G.subgraph(valid_region)))
        if len(comps) > 1:
            largest_comp = max(comps, key=len)
            removed_extras = set().union(
                *[ccmp for ccmp in comps if ccmp != largest_comp])
            valid_region.difference_update(removed_extras)
            unassigned.update(removed_extras)
            removed_set.update(removed_extras)  # Track extras
        attempts += 1

    shortage = target_cardinality - len(valid_region)
    attempts = 0

    # Expansion phase (Fixed)
    while shortage > 0 and attempts < max_attempts:
        # 1. Compute boundary of A_u (BAA_u)
        boundary_Au = [u for u in unassigned if any(
            nb in valid_region for nb in G.neighbors(u))]
        if not boundary_Au:
            break

        # 2. Compute articulation areas of A_u using Trajan's algorithm
        A_u_subgraph = G.subgraph(unassigned)
        articulation_points = set(nx.articulation_points(A_u_subgraph))

        # 3. Filter BAA_u to exclude articulation points (to preserve A_u contiguity)
        BAA_u_filtered = [
            u for u in boundary_Au if u not in articulation_points]
        if not BAA_u_filtered:
            BAA_u_filtered = boundary_Au  # Fallback if all are articulation points

        # 4. Select area to add (prioritize parent areas)
        node_to_add = random.choice(BAA_u_filtered)
        children = parent_map.get(node_to_add, set())
        to_add = {node_to_add} | (children & unassigned)

        # 5. Update region and unassigned
        valid_region.update(to_add)
        unassigned.difference_update(to_add)
        removed_set.difference_update(to_add)  # Avoid re-adding

        # 6. Adjust shortage (w) and handle overshooting
        new_shortage = target_cardinality - len(valid_region)
        if new_shortage < 0:
            # If overshot, check validity
            if len(valid_region) not in allowed_cardinalities:  # Define allowed_cardinalities
                valid_region = set(merged_region)  # Reset
                unassigned.update(removed_set)
                break
        shortage = new_shortage
        attempts += 1

    return valid_region, excess_areas


def build_regions(
        G,
        cardinalities,
        parent_map,
        max_retries: int = 10,
        prrp_variant='PPRP'):

    sorted_cardinalities = sorted(cardinalities, reverse=True)
    unassigned = set(G.nodes())
    regions = []
    seeds = []
    total_iterations = 0
    is_feasible_solution = False

    MS = 10  # Maximum number of iterations (MS)

    # For completeness, return the regions in last iterations, len(regions) = p, that can be used in calculation. Can also max(len(regions)) among all iterations
    for _ in range(MS):
        total_iterations += 1
        regions.clear()
        unassigned = set(G.nodes())
        seeds.clear()
        success = False

        for i, c in enumerate(sorted_cardinalities):
            is_last_region = (i == len(sorted_cardinalities) - 1)
            # Phase 1: Region Growing
            region = region_growing(
                G, c, unassigned, seeds, max_retries=max_retries)
            if not region:
                if is_last_region:
                    region = unassigned
                    unassigned.clear()
                else:
                    break
            if prrp_variant != 'PRRP-G':  # Skip merging and splitting for PRRP-G
                # Phase 2: Region Merging
                merged_region, unassigned = region_merging(
                    G, region, unassigned)
                # Phase 3: Region Splitting (if needed)
                if len(merged_region) > c:
                    valid_region, excess = region_splitting(
                        G, parent_map, merged_region, c, unassigned, cardinalities)
                    unassigned.update(excess)
                else:
                    valid_region = merged_region
            else:
                valid_region = region

            regions.append(valid_region)
            unassigned.difference_update(valid_region)

            # Update seeds with neighbors of the new region
            new_seeds = []
            for node in valid_region:
                new_seeds.extend(
                    [nbr for nbr in G.neighbors(node) if nbr in unassigned])
            seeds.extend(new_seeds)
            seeds = list(set(seeds))

        # Check if the solution is feasible
        if not unassigned:
            is_feasible_solution = True
            return regions, total_iterations, is_feasible_solution
    return regions, total_iterations, is_feasible_solution
