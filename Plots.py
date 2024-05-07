import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio

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
def plot_accuracies(accuracy, name, visualization_path):
    # Graphs the accuracy for each participant or channel compared to random chance (they all beat random chance yay)
    fix, ax = plt.subplots()
    participants = np.linspace(1, len(accuracy), num=len(accuracy))
    ax.bar(participants, accuracy, label="Accuracy")
    ax.set_ylabel("Accuracy")
    x1, y1 = [1, len(accuracy)], [1/3, 1/3]
    ax.plot(x1, y1, 'r', label="Random Chance")
    ax.legend()
    plt.savefig(f'{visualization_path}/{name}')
    plt.clf()