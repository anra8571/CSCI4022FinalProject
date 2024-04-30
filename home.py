from Participant import Participant as p
import numpy as np

# Run each participant as a separate set, getting the clustering accuracy for each and comparing it to the LOOCV accuracy
def run_singles():
    # LOOCV: 0.826667, 0.533333, 0.573333, 0.960000, 0.53333, 0.786667, 0.760000, 0.746667, 0.666667, 0.853333
    # Participant: 1      2         3          4        5         6         7         8         9        10
    accuracies = []

    p1 = p("01")
    p1_clust, p1_acc = p1.cluster()
    accuracies.append(p1_acc)

    p2 = p("02")
    p2_clust, p2_acc = p2.cluster()
    accuracies.append(p2_acc)

    p3 = p("03")
    p3_clust, p3_acc = p3.cluster()
    accuracies.append(p3_acc)

    p4 = p("04")
    p4_clust, p4_acc = p4.cluster()
    accuracies.append(p4_acc)

    p5 = p("05")
    p5_clust, p5_acc = p5.cluster()
    accuracies.append(p5_acc)

    p6 = p("06")
    p6_clust, p6_acc = p6.cluster()
    accuracies.append(p6_acc)

    p7 = p("07")
    p7_clust, p7_acc = p7.cluster()
    accuracies.append(p7_acc)

    p8 = p("08")
    p8_clust, p8_acc = p8.cluster()
    accuracies.append(p8_acc)

    p9 = p("09")
    p9_clust, p9_acc = p9.cluster()
    accuracies.append(p9_acc)

    p10 = p("10")
    p10_clust, p10_acc = p10.cluster()
    accuracies.append(p10_acc)

    participant_array = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]

    print(f"Accuracies for the {len(participant_array)} participants: {accuracies}")

    return participant_array

# Input: participants is an array of all the participants loaded
# Input: channel is an integer between 1-40 indicating which channel to pull
def get_channel(participants, channel):
    channel_array = []
    labels_array = []

    for p in participants:
        for trial in range(0, len(p.fv)):
            # Account for Python being 0-indexed
            channel_array.append(p.fv[trial, channel - 1, :])
        for elem in p.labels:
            labels_array.append(elem)

    return np.array(channel_array), np.array(labels_array)

def get_all_participants(participants):
    p_array = []
    labels_array = []

    for p in participants:
        for trial in range(0, len(p.fv)):
            p_array.append(p.fv[trial, :, :])
        for elem in p.labels:
            labels_array.append(elem)

    return np.array(p_array), np.array(labels_array)

if __name__ == "__main__":
    # Gets array of all the individual participants clustered by trial
    ind_participants = run_singles()

    # A Participant containing data for channel 1 across all participants, all trials
    c1_array, c1_labels = get_channel(np.array(ind_participants), 1)
    c1 = p("c1", fv=c1_array, labels=c1_labels)

    # A Participant containing all trials, all channels, all participants. Cluster by trial
    all_array, labels_array = get_all_participants(ind_participants)
    labels_array.reshape([1, len(ind_participants) * 75])
    all = p("all", fv=all_array, labels=labels_array)
    all.labels = labels_array
    all_clusters, all_accuracy = all.cluster()