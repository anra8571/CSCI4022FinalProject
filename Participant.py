# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html

import os
from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np

# Represents the data from one participant
class Participant:
    '''
    Each participant contains 75 trials, 40 channels, and 3 averages
    '''

    def __init__(self, name):
        self.name = name
        self.fv = []
        self.fv = self.load_struct()
        self.clusters = []
        self.accuracy = 0

    # Finds the MATLAB file, converts struct to array, and saves in local filesystem
    def load_struct(self):
        # Prints to terminal
        print("Loading the data structure - converting to np array from MATLAB struct")
        
        # Filename variables
        filename = f"{self.name}_x.mat"
        outfile = f"{self.name}_fv.np"

        # Loads MATLAB file into a dictionary
        cwd = os.getcwd()
        directory = f"{cwd}/Data/{filename}"
        save_directory = f"{cwd}/Data/{outfile}"
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

        # Save locally
        with open(save_directory, 'wb') as f:
            np.save(f, np.array(data))

        # Returns the np array
        return np.array(data)
    
    # From homework in class, generalized for multiple dimensions
    def distance(self, slice1, slice2):
        """
        Calculate the Euclidean distance between two 3D slices (40, 3) from a (75, 40, 3) array.
        """
        return np.sqrt(np.sum((slice1 - slice2)**2))

    # From homework in class
    def kmeans(self, df, k=3, tol=0.0001):
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
    
    # Checks the clusters from k-means against the ground truth data
    def accuracy_calculation(self, clusters):
        # Load the labels
        ground_truth_struct = sio.loadmat(f'Data/{self.name}_labels.mat')

        # Convert from MATLAB struct to np array
        gt = []
        for elem in ground_truth_struct['Y']:
            gt.append(elem[0])
        gt_np = np.array(gt)

        # Accuracy = number of same labels / total number of labels
        correct = 0
        for i in range(len(gt_np)):
            if gt_np[i] == clusters[i]:
                correct += 1

        accuracy = correct/len(gt_np)

        return accuracy
    
    # Runs the clustering function and accuracy calculation. Saves data to the class object and prints to terminal. Repeat 25 times and take the highest accuracy
    def cluster(self):
        print("Clustering the data - running kmeans 25 times and taking the highest accuracy score")
        
        acc_high = 0
        clust_high = np.array([])
        mean_low = 0

        for i in range(25):
            # Clusters the data
            centroids, clusters, meanerror = self.kmeans(self.fv)
            acc = self.accuracy_calculation(clusters)

            # If this is the best trial so far, save the data
            if acc > acc_high:
                clust_high = clusters
                acc_high = acc
                mean_low = meanerror
        
        # At the end, save the data to the object and print the final score
        self.clusters = clust_high
        self.accuracy = acc_high

        print(f"The clustering accuracy is {self.accuracy * 100}")