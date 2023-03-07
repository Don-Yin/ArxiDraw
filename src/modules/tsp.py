from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import multiprocessing as mp
import time
import queue
import numpy as np
import threading
from numba import njit, prange
from tqdm import tqdm
from matplotlib import pyplot as plt
import random


def cpaste(test: str, color: str):
    """
    Color paste text to terminal outpu, for example: print(f"{cpaste('hello',
    'red')} world") will print hello in red
    ---------------------------------------------------------------------------
    """
    color_dict = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "black": "\033[98m",
        "end": "\033[0m",
    }
    return f"{color_dict[color]}{test}{color_dict['end']}"


def create_locations(num, width):
    return np.random.uniform(0, width, (num, 2))


def plot_cycle(locs, cycle):
    x, y = zip(*locs)
    plt.figure()
    plt.scatter(x, y)

    for idx in range(cycle.shape[0]):
        x0, y0 = locs[cycle[idx]]
        x1, y1 = locs[cycle[(idx + 1) % locs.shape[0]]]
        plt.plot([x0, x1], [y0, y1], c="r", linewidth=1)


@njit  # the parallel decorator won't work here because this function will be part of a multiprocessing pool
def cycle_length_fast(locs, cycle):
    """
    A numba implementation of the cycle_length function that uses a loop
    instead of numpy.cumsum as the latter is not supported by numba.
    ---------------------------------------------------------------------------
    params:
        locs (np.ndarray): the input array
        cycle (np.ndarray): the input array
    returns:
        np.ndarray: the result array
    """
    result = 0
    for i in prange(len(cycle)):
        x1, y1 = locs[cycle[i]]
        x2, y2 = locs[cycle[(i + 1) % len(cycle)]]
        result += ((x2 - x1) ** 2) + ((y2 - y1) ** 2) ** 0.5
    return result


@njit(nogil=True, fastmath=True)  # add no gil for later use of threading
def generate_mutation_fast(cycle, num_changes):
    """
    Given a cycle, apply `num_changes` mutations to the cycle and return the
    mutant. This is a faster version of the generate_mutation function. It uses
    the random.randint function instead of np.random.randint for cases where
    only a single value is needed.
    ---------------------------------------------------------------------------
    params:
        cycle (np.ndarray): the cycle
        num_changes (int): the number of changes to make
    returns:
        np.ndarray: the mutated cycle
    """
    cycle_copy = cycle.copy()
    # for a single value, this is faster than np.random.randint
    num_mutations = random.randint(1, num_changes + 1)
    # preallocate the indices array to avoid the overhead of creating a new array each time
    indices = np.random.randint(0, len(cycle_copy), size=(num_mutations, 2))
    for i, j in indices:
        cycle_copy[i], cycle_copy[j] = cycle_copy[j], cycle_copy[i]
    return cycle_copy


def find_better_cycle(locs, init_cycle, num_iterations, num_offspring, num_changes):
    """
    This is a modified version of the genetic algorithm that is used to find a
    better cycle. The following changes were made:
        - The cycle length is calculated using the numba implementation
        - The mutation is calculated using the numba implementation
        - an early stopping condition was added
        - last improvement is calculated using numpy
        - np.argmin is used to find the index of the best cycle
    ---------------------------------------------------------------------------
    params:
        locs (np.array): locations of the points
        init_cycle (np.array): initial cycle of the points
        num_iterations (int): number of iterations to run the genetic algorithm
        num_offspring (int): number of offspring to generate per iteration
        num_changes (int): number of changes to make to the cycle per mutation
    returns:
        current_cycle (np.array): best cycle found
        iter_lengths (np.array): array of cycle lengths for each iteration
        num_iterations (int): number of iterations run
    """
    current_cycle = init_cycle
    iter_lengths = []

    for i in range(num_iterations):
        try:
            # find the number of iterations since the last improvement
            num_last_improvement = len(iter_lengths) - np.argmin(iter_lengths) - 1
        except ValueError:
            # cases where the first iteration is the best
            num_last_improvement = 0

        offspring = [generate_mutation_fast(current_cycle, num_changes) for _ in range(num_offspring)]

        family = [current_cycle] + offspring

        # the threading trick above does not make the lengths faster so it is not used here
        lengths = np.array([cycle_length_fast(locs, c) for c in family])

        idx = np.argmin(lengths)
        current_cycle = family[idx]
        iter_lengths.append(lengths[idx])

        # early stopping condition if the best cycle has not improved in 512 iterations
        if num_last_improvement > 512:
            return current_cycle, iter_lengths, i

    return current_cycle, iter_lengths, num_iterations


def attach_cluster(to_locs, from_locs, from_cycle):
    """
    This function takes two clusters and tries to attach them together. This
    is for attaching adjacent clusters when breaking the locations into
    clusters.
    ---------------------------------------------------------------------------
    params:
        from_locs: locations of the first cluster
        from_cycle: cycle of the first cluster
        to_locs: locations of the second cluster
    """
    # the the coordinates of the first element of the to cycle
    to_point = to_locs[0]
    # find the index of the element in the from_cycle that is closest to the to_point
    from_point_index = np.argmin(np.linalg.norm(from_locs - to_point, axis=1))
    # roll the from_cycle so that the from_point_index is the first element
    from_cycle = np.roll(from_cycle, -from_point_index)
    return from_cycle


class GeneticAlgorithm:
    def __init__(self, size_locs, total_num_iterations):
        """
        The GeneticAlgorithm class contains methods for implementing a genetic
        algorithm to solve the Traveling Salesman Problem. Its main method, main(),
        takes in locations (locs), current cycle of the points, number of locations
        per cluster, and total number of iterations. It breaks the locations into
        clusters and runs the find_better_cycle method in parallel on each cluster
        locally using the multiprocessing library. The find_better_cycle method
        uses a genetic algorithm to find a better cycle for the points in a given
        cluster. The main method also updates a progress bar to track the progress
        of the algorithm. The break_to_clusters method is a helper function that
        breaks the locations into clusters and returns the clusters and their
        order.
        ---------------------------------------------------------------------------
        params:
            size_locs (int): the number of locations for progress bar
            total_num_iterations (int): the total number of iterations for the
                algorithm
        """
        self.total_num_batches = 0
        self.main_cycle = True
        self.size_locs = size_locs  # initial record of the size of the locs array for progress bar
        self.total_num_iterations = total_num_iterations  # initial record of the total number of iterations for progress bar

    def main(self, locs, cycle, num_locs_per_cluster):
        """
        The main method of the GeneticAlgorithm class performs the following actions:
            1. It divides the input locs into clusters.
            2. It runs the find_better_cycle method in parallel on each cluster using
                the multiprocessing library.
            3. If the number of clusters is greater than the input num_locs_per_cluster,
                the method further divides the clusters into smaller clusters in a loop
                until the number of clusters is less than or equal to num_locs_per_
                cluster.
            4. It combines the clusters by rolling them so that the first element of
                each cluster is the closest element to the first element of the
                previous cluster.
            5. It returns the modified cycle and locs.
        ---------------------------------------------------------------------------
        params:
            locs (np.array): locations of the points
            cycle (np.array): current cycle of the points
            num_locs_per_cluster (int):
                number of locations per cluster, minimum 2. The smaller the number
                the more clusters will be created and more accurate the result will be.
                But the computation time will increase exponentially.
        returns:
            cycle (np.array): new cycle of the points
            locs (np.array): new locations of the points
        """
        # break the locations into clusters
        num_clusters = np.ceil(len(locs) / num_locs_per_cluster).astype(int)
        # calculate the total number of batches
        self.total_num_batches += np.ceil(num_clusters / mp.cpu_count()).astype(int)
        # break the locations into clusters
        locs_clusters, cycle_clusters, cluster_order = self.break_to_clusters(locs, cycle, num_clusters, num_locs_per_cluster)
        # run the find_better_cycle method in parallel on each cluster
        num_iterations_per_cluster = self.total_num_iterations // num_clusters

        # if this is the first time running the main method, then print the number of batches and the number of cores
        if self.main_cycle:
            print("-" * 2**6)
            print(
                f"Total number of batches: {cpaste(self.total_num_batches, 'yellow')} | Locs size: {cpaste(self.size_locs, 'yellow')} | Using {cpaste(mp.cpu_count(), 'yellow')} cores"
            )
            self.progress_bar = tqdm(range(self.total_num_batches), colour="green")
            self.batches_iter = iter(range(self.total_num_batches))
            self.main_cycle = False

        # init the results array
        results = []

        # calculate the number of batches by dividing the number of clusters by the number of cores
        num_batches = np.ceil(num_clusters / mp.cpu_count()).astype(int)

        for batch in range(num_batches):
            with mp.Pool(mp.cpu_count()) as pool:  # use all available cores
                results += pool.starmap(
                    find_better_cycle,
                    [
                        (
                            locs_clusters[cluster],
                            np.arange(len(locs_clusters[cluster])),
                            num_iterations_per_cluster,
                            2**6,
                            2**1,  # set a low mutation rate as the clusters are already close to the optimal solution
                        )
                        for cluster in range(batch * mp.cpu_count(), min((batch + 1) * mp.cpu_count(), num_clusters))
                    ],
                )
            current_batch = self.batches_iter.__next__()
            iterations_ran = sum((result[2] for result in results))
            self.progress_bar.set_description(
                f"Batch {cpaste(current_batch + 1, 'red')} | Planned Iterations: {cpaste(int(self.total_num_iterations / self.total_num_batches * (current_batch + 1)), 'red')} | Iterations ran (early stopping): {cpaste(iterations_ran, 'red')}"
            )
            self.progress_bar.update(1)

        new_cycle = [i[0] for i in results]
        new_cycle = [cycle_clusters[cluster][new_cycle[cluster]] for cluster in cluster_order]
        for cluster in range(num_clusters - 1):
            new_cycle[cluster + 1] = attach_cluster(
                locs_clusters[cluster_order[cluster]],
                locs_clusters[cluster_order[cluster + 1]],
                new_cycle[cluster + 1],
            )
        new_cycle = np.concatenate(new_cycle)

        # finally run the find_better_cycle method on the entire set of locations
        # to make sure remove any local minima
        new_cycle, _, _ = find_better_cycle(locs, new_cycle, 999999, 2**6, 2**1)

        # calculate the length of the new cycle
        length = cycle_length_fast(locs, new_cycle)

        return new_cycle, length

    def break_to_clusters(self, locs, cycle, num_clusters, num_locs_per_cluster):
        """
        This function breaks the locations into clusters and returns the clusters
        and the order of the clusters. This is for breaking the locations into
        clusters before running the genetic algorithm. This is done to reduce
        the number of locations in the genetic algorithm.

        params:
            locs (np.array): locations of the points
            cycle (np.array): cycle of the points
            num_clusters (int): number of clusters to break the locations into
            num_locs_per_cluster (int): number of locations per cluster
        returns:
            locs_clusters (np.array): array of arrays of locations of the clusters
            cycle_clusters (np.array): array of arrays of cycles of the clusters
            cluster_order (np.array): order of the clusters
        """
        # scale the coordinates to be between 0 and 1 and then cluster them
        scaler = StandardScaler()
        locs_scaled = scaler.fit_transform(locs)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(locs_scaled)
        locs_clusters_index = kmeans.predict(locs_scaled)

        # divide the coordinates into clusters from 0 to max (array of arrays of a size 8)
        locs_clusters = np.array([locs[locs_clusters_index == cluster] for cluster in range(num_clusters)], dtype=object)
        cycle_clusters = np.array([cycle[locs_clusters_index == cluster] for cluster in range(num_clusters)], dtype=object)
        centers_locs = np.array([np.mean(locs_clusters[cluster], axis=0) for cluster in range(num_clusters)])

        # find the order of the clusters by viewing them as a graph and finding the shortest path
        # if the number of clusters is less than the number of locations per cluster, then
        # just feed back the original cycle for further dimensionality reduction, otherwise
        # find the order of the clusters
        centers_cycle = np.arange(len(centers_locs))
        if (
            len(centers_locs) > num_locs_per_cluster
        ):  # if the number of clusters is greater than the number of locations per cluster further reduce the number of clusters
            cluster_order, _ = self.main(centers_locs, centers_cycle, num_locs_per_cluster)
        else:
            cluster_order, _, _ = find_better_cycle(centers_locs, centers_cycle, 999999, 2**6, 2**1)
        return locs_clusters, cycle_clusters, cluster_order


"""On my machine the njit will sometimes fails to compile without an error message (5% chance). If that happens, please try running the cell again."""
from modules.tsp import GeneticAlgorithm, cpaste, create_locations, plot_cycle
import time
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # show the result for different number of locations
    loc_sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    for i in loc_sizes:
        locs = create_locations(i, 10)
        cycle = np.arange(i)
        # define the number of locations per cluster
        # number of locations per cluster, minimum 2. The smaller the number the more clusters will be created and more accurate the result will be. But the computation time will increase exponentially. So for large number of locations, it is better to keep the number of locations relatively larger. For small numbers of locations, we can afford to keep the number of locations per cluster smaller.
        match i:
            case 100:
                num_locs_per_cluster = 2**2  # smaller = more clusters = more accurate = more time
            case 1_000:
                num_locs_per_cluster = 2**2
            case 10_000:
                num_locs_per_cluster = 2**4
            case 100_000:
                num_locs_per_cluster = 2**7
            case 1_000_000:  # at my 16 core machine, this takes about 13 minutes to run
                num_locs_per_cluster = 2**8  # smaller = more clusters = more accurate = more time
        tval = time.time()
        process_ = GeneticAlgorithm(len(locs), total_num_iterations=2000000)
        new_cycle, length = process_.main(locs, cycle, num_locs_per_cluster)
        tval = time.time() - tval
        # because the map is divided and processed as many small local batches, we don't obtain the semilogy plot here (since the global length is not computed at each batch) as we did in the previous case. But we can still plot the cycle and see the result.
        print(f"↓ Time: {cpaste(tval.__round__(3), 'green')} seconds, Best length: {cpaste(length.__round__(3), 'green')} ↓")
        plot_cycle(locs, new_cycle)
        plt.show()
