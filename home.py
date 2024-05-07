# Import other files
from Participant import Participant as p
import DataRetrieval
import Plots
import MLP
import DimReduction

# Import libraries
import numpy as np # type: ignore

# Paths
visualization_path = './Visualizations/'

# Parameters
num_participants = 30
num_trials = 75
k = 25

# Printing Options - if toggled, prints output to file
# np.set_printoptions(threshold=sys.maxsize)
# sys.stdout = open('output.txt', 'wt')

# Automates loading of data, all types of clustering in report, and data visualizations
if __name__ == "__main__":
    # Gets array of all the individual participants clustered by trial
    ind_participants, ind_accuracies, rands = DataRetrieval.run_singles()

    # Perform dimension reduction on each participant, then cluster again and get the accuracy
    dim_red_acc = []
    for part in ind_participants:
        part.dim_red = DimReduction.feature_reduction(part.fv_reduced)
        c0lust, acc, rand = part.cluster(dim=2)
        dim_red_acc.append(acc)

    # Plot the accuracy of the original data vs. the accuracy after dimension reduction
    Plots.plot_compare(ind_accuracies, dim_red_acc, visualization_path)

    # Graphs the accuracy for each partcipant compared to random chance (they all beat random chance yay)
    Plots.plot_accuracies(ind_accuracies, 'Accuracy by Individual Participant', visualization_path)

    # A Participant containing data for first 20 channels across all participants, all trials
    channels_array, channels_accuracy = DataRetrieval.run_channels(np.array(ind_participants))

    # Graphs the accuracy by channel
    Plots.plot_accuracies(channels_accuracy, 'Accuracies by Individual Channel', visualization_path)

    # A Participant containing all trials, all channels, all participants. Cluster by trial
    all_array, all_reduced_array, labels_array = DataRetrieval.get_all_participants(ind_participants)
    all = p("all", fv=all_array, fv_reduced=all_reduced_array, labels=labels_array)
    all.labels = labels_array
    all_clusters, all_accuracy, all_rands = all.cluster(k)
    all_mlp_acc = MLP.MLP(all)
    Plots.plot_compare_aggregate([all_accuracy], [all_mlp_acc], "Kmeans Clustering vs Multi Layer Perceptron", 'K means', 'MLP', visualization_path)

    # Plot the accuracy for all participants by activity
    Plots.plot_by_activity(ind_participants[0], f'Accuracy for {ind_participants[0].name} by Activity', visualization_path, num_trials)
    Plots.plot_by_activity(all, 'Accuracy by Activity Type', visualization_path, num_trials)