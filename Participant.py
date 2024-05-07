# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration
# https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

import MLP
import os
import scipy.io as sio  # type: ignore
import numpy as np # type: ignore
from scipy import stats # type: ignore
from sklearn import metrics # type: ignore

# Represents the data from one participant
class Participant:
    '''
    Each participant contains 75 trials, 40 channels, and 3 averages
    '''

    def __init__(self, name, fv=[], fv_reduced=[], labels=None):
        self.name = name
        self.fv = fv # Feature vector in 3D np array
        self.fv_reduced = fv_reduced # Feature vector in 2D array
        self.dim_red = 0 # Dimension reduction using PCA

        self.clusters = []
        self.accuracy = 0
        self.rand_index = 0
        self.mlp = 0

        # Some datasets are combinations of participants, in which the feature vector/labels is externally constructed and passed in as an argument
        if len(self.fv) == 0:
            self.fv, self.fv_reduced = self.load_struct()
        if labels is None:
            self.labels = self.get_labels()
        else:
            self.labels = labels

    # Get the ground truth labels from the MATLAB file
    def get_labels(self):
        # Load the labels
        ground_truth_struct = sio.loadmat(f'Data/Clustering/{self.name}_labels.mat')

        # Convert from MATLAB struct to np array
        gt = []
        for elem in ground_truth_struct['Y']:
            gt.append(elem[0])
        gt_np = np.array(gt)

        return gt_np

    # Finds the MATLAB file, converts struct to array, and saves in local filesystem
    def load_struct(self):
        # Prints to terminal
        print("Loading the data structure - converting to np array from MATLAB struct")
        
        # Filename variables
        filename = f"{self.name}_x.mat"
        d3_outfile = f"{self.name}_fv.np"
        d2_outfile = f"{self.name}_fv_reduced.np"

        # Loads MATLAB file into a dictionary
        cwd = os.getcwd()
        directory = f"{cwd}/Data/Clustering/{filename}"
        save_directory_3d = f"{cwd}/Data/NP/{d3_outfile}"
        save_directory_2d = f"{cwd}/Data/NP/{d2_outfile}"
        mat_content = sio.loadmat(directory)["X"]

        # Reads in 2D MATLAB struct and converts to expected 3D Numpy array (in MATLAB, the epoch and channel are both contained in column)
        col_count = 0
        data = []
        trial = []
        for row in range(0, len(mat_content)):
            epoch = []
            for col in range(0, len(mat_content[0])):
                # Every set of three elements represents a single feature
                if col_count < 2:
                    epoch.append(mat_content[row, col])
                    col_count += 1
                else:
                    epoch.append(mat_content[row, col])
                    trial.append(epoch)
                    epoch = []
                    col_count = 0
            data.append(trial)
            trial = []

        # Reads in 2D MATLAB struct and saves as original 2D format (rows: trials, columns: channels and epochs)
        two_data = [] # Contains full set
        for row in range(0, len(mat_content)):
            trial = []
            for col in range(0, len(mat_content[0])):
                trial.append(mat_content[row, col])
            two_data.append(trial)
            trial = []

        # Save locally
        with open(save_directory_3d, 'wb') as f:
            np.save(f, np.array(data))

        with open(save_directory_2d, 'wb') as f:
            np.save(f, np.array(two_data))

        # Returns the np array
        return np.array(data), np.array(two_data)
    
    # From homework in class, generalized for multiple dimensions
    def distance(self, slice1, slice2):
        """
        Calculate the Euclidean distance between two 3D slices (40, 3) from a (75, 40, 3) array.
        """
        return np.sqrt(np.sum((slice1 - slice2)**2))

    # From homework in class
    def kmeans_3D(self, df, k=3, tol=0.0001):
        """
        K-means clustering for 3D data.

        Parameters:
            df (numpy.ndarray): 3D numpy array with shape (75, 40, 3), each slice (40, 3) treated as a point.
            k (int): Number of clusters.
            tol (float): Tolerance for L_2 convergence check on centroids.

        Returns:
            centroids (numpy.ndarray): Array of centroids, one for each cluster.
            clusters (numpy.ndarray): Cluster assignment for each point.
            rec_error (float): Reconstruction error on final iteration.
        """    
        # Initialize reconstruction error for 1st iteration
        prev_rec_error = np.inf
        
        # Random centroids from data
        clocs = np.random.choice(df.shape[0], size=k, replace=False)
        centroids = df[clocs, :, :].copy()
        
        # Initialize objects for points-cluster distances and cluster assignments.
        dists = np.zeros((k, df.shape[0]))
        clusters = np.array([-1] * df.shape[0])
        
        # Index and convergence trackers
        ii = 0
        Done = False
        while not Done:
            # Update classifications
            for ji in range(k):
                for pi in range(df.shape[0]):
                    dists[ji, pi] = self.distance(df[pi, :, :], centroids[ji, :, :])
            
            clusters = dists.argmin(axis=0)
            
            # Update centroids
            for ji in range(k):
                if np.sum(clusters == ji) > 0:
                    centroids[ji, :, :] = np.mean(df[clusters == ji], axis=0)
                else:
                    # Reinitialize centroid if no points are assigned to prevent empty clusters
                    centroids[ji, :, :] = df[np.random.choice(df.shape[0], size=1), :, :]

            # Calculate Reconstruction Error    
            rec_error = np.sum(np.min(dists, axis=0)**2) / df.shape[0]
            
            # Convergence check
            change_in_error = np.abs(prev_rec_error - rec_error)
            if change_in_error < tol:
                # print(f'Done at iteration {ii} with change of {change_in_error}')
                Done = True
            elif ii == 50:
                # print('No convergence in 50 steps')
                Done = True
            
            prev_rec_error = rec_error
            ii += 1
        
        # The MATLAB data is 1-indexed, so we add 1 here so we can compare for accuracy
        clusters += 1

        return centroids, clusters, rec_error
    
    # From homework in class
    def kmeans_2D(self, df, k=3, tol=0.0001):
        # Initialize reconstruction error for 1st iteration
        prev_rec_error = np.inf
        
        # Random centroids from data
        clocs = np.random.choice(df.shape[0], size=k, replace=False)
        centroids = df[clocs, :].copy()
        
        # Initialize objects for points-cluster distances and cluster assignments.
        dists = np.zeros((k, df.shape[0]))
        clusters = np.array([-1] * df.shape[0])
        
        # Index and convergence trackers
        ii = 0
        Done = False
        while not Done:
            # Update classifications
            for ji in range(k):
                for pi in range(df.shape[0]):
                    dists[ji, pi] = self.distance(df[pi, :], centroids[ji, :])
            
            clusters = dists.argmin(axis=0)
            
            # Update centroids
            for ji in range(k):
                if np.sum(clusters == ji) > 0:
                    centroids[ji, :] = np.mean(df[clusters == ji], axis=0)
                else:
                    # Reinitialize centroid if no points are assigned to prevent empty clusters
                    centroids[ji, :] = df[np.random.choice(df.shape[0], size=1), :]

            # Calculate Reconstruction Error    
            rec_error = np.sum(np.min(dists, axis=0)**2) / df.shape[0]
            
            # Convergence check
            change_in_error = np.abs(prev_rec_error - rec_error)
            if change_in_error < tol:
                # print(f'Done at iteration {ii} with change of {change_in_error}')
                Done = True
            elif ii == 50:
                # print('No convergence in 50 steps')
                Done = True
            
            prev_rec_error = rec_error
            ii += 1
        
        # The MATLAB data is 1-indexed, so we add 1 here so we can compare for accuracy
        clusters += 1

        return centroids, clusters, rec_error
    
    # Checks the clusters from k-means against the ground truth data
    # Input: Clusters (predicted group)
    # Input: Labels (ground truth)
    def accuracy_calculation(self, clusters, labels):
        if labels is None:
            labels  # type: ignore

        # Accuracy = number of same labels / total number of labels
        correct = 0
        one_c = 0
        one_total = 0
        two_c = 0
        two_total = 0
        three_c = 0
        three_total = 0

        for i in range(len(self.labels)):
            if self.labels[i] == clusters[i]:
                correct += 1
                if self.labels[i] == 1:
                    one_c += 1
                elif self.labels[i] == 2:
                    two_c += 1
                elif self.labels[i] == 3:
                    three_c += 1
            if self.labels[i] == 1:
                one_total += 1
            elif self.labels[i] == 2:
                two_total += 1
            elif self.labels[i] == 3:
                three_total += 1

        accuracy = correct/len(self.labels)
        one_acc = one_c / one_total
        two_acc = two_c / two_total
        three_acc = three_c / three_total

        rand_score = metrics.adjusted_rand_score(self.labels, clusters)

        return accuracy, one_acc, two_acc, three_acc, rand_score
    
    # Runs the clustering function and accuracy calculation. Saves data to the class object and prints to terminal. Repeat 25 times and take the highest accuracy
    def cluster(self, k=25, labels=None, dim=3):
        print(f"Clustering the data for {self.name} - running kmeans {k} times and taking the highest accuracy score")

        acc_high = 0
        clust_high = np.array([])
        mean_low = 0

        for i in range(k):
            # Clusters the data
            if dim == 3:
                centroids, clusters, meanerror = self.kmeans_3D(self.fv)
            # If specified, clusters the data using dimension reduction
            else:
                centroids, clusters, meanerror = self.kmeans_2D(self.dim_red)
            acc, one, two, three, rand_ind = self.accuracy_calculation(clusters, labels)

            # If this is the best trial so far, save the data
            if acc > acc_high:
                clust_high = clusters
                acc_high = acc
                mean_low = meanerror
        
        # At the end, save the data to the object and print the final score
        self.clusters = clust_high
        self.accuracy = acc_high
        self.rand_index = rand_ind

        # Testing against a different method to see if k-means holds up to other clustering algorithms
        try:
            self.mlp = MLP.MLP(self)
            print(f"The k-means clustering accuracy for {self.name} is {self.accuracy * 100} and the Rand index is {self.rand_index} and the MLP score is {self.mlp}")
        except ValueError:
            print("The data is not in the right shape for MLP")

        return self.clusters, self.accuracy, self.rand_index

    # def hierarchical_cluster(self):
    #     agglo = cluster.FeatureAgglomeration(n_clusters = 3)
    #     agglo.fit(self.fv)

    #     print("Agglo: ", agglo)

    #     fv_reduced = agglo.transform(self.fv)
    #     print("FV Reduced: ", fv_reduced)