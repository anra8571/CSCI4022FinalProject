# Import other files
from Participant import Participant as p
import DataRetrieval
import Plots

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio

# Paths
visualization_path = './Visualizations/'

# Parameters
num_participants = 30
num_trials = 75
k = 25

# Printing Options - if toggled, prints output to file
# np.set_printoptions(threshold=sys.maxsize)
# sys.stdout = open('output.txt', 'wt')

if __name__ == "__main__":
    # Gets array of all the individual participants clustered by trial
    ind_participants, ind_accuracies = DataRetrieval.run_singles()

    # Graphs the accuracy for each partcipant compared to random chance (they all beat random chance yay)
    Plots.plot_accuracies(ind_accuracies, 'Accuracies by Individual Participant', visualization_path)

    # A Participant containing data for first 20 channels across all participants, all trials
    channels_array, channels_accuracy = DataRetrieval.run_channels(np.array(ind_participants))

    # Graphs the accuracy by channel
    Plots.plot_accuracies(channels_accuracy, 'Accuracies by Individual Channel', visualization_path)

    # A Participant containing all trials, all channels, all participants. Cluster by trial
    all_array, labels_array = DataRetrieval.get_all_participants(ind_participants)
    all = p("all", fv=all_array, labels=labels_array)
    all.labels = labels_array
    all_clusters, all_accuracy = all.cluster(k)

    # # Plot the accuracy for all participants by activity
    Plots.plot_by_activity(ind_participants[0], f'Accuracy for {ind_participants[0].name} by Activity', visualization_path, num_trials)
    Plots.plot_by_activity(all, 'Accuracy by Activity Type', visualization_path, num_trials)