# For help with the comparison bar chart: https://stackoverflow.com/questions/53182452/python-create-bar-chart-comparing-2-sets-of-data
# Stacked bar charts: https://stackoverflow.com/questions/14270391/how-to-plot-multiple-bars-grouped

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def visualize(x, y, cluster):
    plt.colorbar(plt.scatter(x, y, c=cluster))
    plt.legend()
    plt.xlabel("Trial Number")
    plt.ylabel("Participant Number")
    plt.show()

# Plots the accuracy by activity
def plot_by_activity(participant, name, visualization_path, num_trials=75):
    nums = ["Right-Hand Tapping","Left-Hand Tapping","Foot Tapping"]
    labels = participant.labels
    clusters = participant.clusters

    # Counting accuracy
    count1_right = 0
    count2_right = 0
    count3_right = 0

    for i in range(num_trials):
        if(labels[i]==1 and clusters[i]==1): count1_right += 1
        elif(labels[i]==2 and clusters[i]==2): count2_right += 1
        elif(labels[i]==3 and clusters[i]==3): count3_right += 1
    
    # The study divides the activities evenly - each participant should have 25 right-hand, 25 left-hand, and 25 foot-tapping
    count_arr = [count1_right/.25, count2_right/.25, count3_right/.25]
    print(f"Accuracies by Activity for {participant.name}: {count_arr}")
    plt.bar(nums, count_arr, alpha=0.7)
    plt.xlabel("Activity")
    plt.ylabel("Accuracy")
    plt.savefig(f"{visualization_path}/{name}")
    plt.clf()

# Plots accuracy by participant or channel
def plot_accuracies(accuracy, name, visualization_path, random_chance=True):
    # Graphs the accuracy for each participant or channel compared to random chance (they all beat random chance yay)
    fix, ax = plt.subplots()
    participants = np.linspace(1, len(accuracy), num=len(accuracy))
    ax.bar(participants, accuracy, label="Accuracy")
    ax.set_ylabel("Accuracy")
    x1, y1 = [1, len(accuracy)], [1/3, 1/3]
    if random_chance:
        ax.plot(x1, y1, 'r', label="Random Chance")
    ax.legend()
    plt.savefig(f'{visualization_path}/{name}')
    plt.clf()

# Plots two sets of data side-by-side for comparison, bar chart, one aggregate
def plot_compare_aggregate(set_1, set_2, name, l1, l2, visualization_path):
    ind = np.arange(1)
    bar_width = .3
    fig, ax = plt.subplots()
    g1 = ax.bar(ind, set_1, bar_width)
    g2 = ax.bar(ind + bar_width, set_2, bar_width)
    ax.legend((g1[0], g2[0]), ("Normal", "Reduced Dimensions"))
    plt.savefig(f'{visualization_path}/Reduced Dimension Comparison')
    plt.clf()

# Plots two sets of data side-by-side for comparison, bar chart, for all participants
def plot_compare(y1, y2, visualization_path):
    ind = np.arange(30)
    bar_width = .3
    fig, ax = plt.subplots()
    g1 = ax.bar(ind, y1, bar_width)
    g2 = ax.bar(ind + bar_width, y2, bar_width)
    ax.legend((g1[0], g2[0]), ("Normal", "Reduced Dimensions"))
    plt.savefig(f'{visualization_path}/Reduced Dimension Comparison')
    plt.clf()