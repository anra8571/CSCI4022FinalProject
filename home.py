from Participant import Participant as p
import numpy as np
import matplotlib.pyplot as plt

# Run each participant as a separate set, getting the clustering accuracy for each and comparing it to the LOOCV accuracy
def run_singles():
    # LOOCV: 0.826667, 0.533333, 0.573333, 0.960000, 0.53333, 0.786667, 0.760000, 0.746667, 0.666667, 0.853333
    # Participant: 1      2         3          4        5         6         7         8         9        10
    accuracies = []

    p1 = p("01")
    p1_clust, p1_acc = p1.cluster_3D()
    accuracies.append(p1_acc)

    p2 = p("02")
    p2_clust, p2_acc = p2.cluster_3D()
    accuracies.append(p2_acc)

    p3 = p("03")
    p3_clust, p3_acc = p3.cluster_3D()
    accuracies.append(p3_acc)

    p4 = p("04")
    p4_clust, p4_acc = p4.cluster_3D()
    accuracies.append(p4_acc)

    p5 = p("05")
    p5_clust, p5_acc = p5.cluster_3D()
    accuracies.append(p5_acc)

    p6 = p("06")
    p6_clust, p6_acc = p6.cluster_3D()
    accuracies.append(p6_acc)

    p7 = p("07")
    p7_clust, p7_acc = p7.cluster_3D()
    accuracies.append(p7_acc)

    p8 = p("08")
    p8_clust, p8_acc = p8.cluster_3D()
    accuracies.append(p8_acc)

    p9 = p("09")
    p9_clust, p9_acc = p9.cluster_3D()
    accuracies.append(p9_acc)

    p10 = p("10")
    p10_clust, p10_acc = p10.cluster_3D()
    accuracies.append(p10_acc)

    p11 = p("11")
    p11_clust, p11_acc = p11.cluster_3D()
    accuracies.append(p11_acc)

    p12 = p("12")
    p12_clust, p12_acc = p12.cluster_3D()
    accuracies.append(p12_acc)

    p13 = p("13")
    p13_clust, p13_acc = p13.cluster_3D()
    accuracies.append(p13_acc)

    p14 = p("14")
    p14_clust, p14_acc = p14.cluster_3D()
    accuracies.append(p14_acc)

    participant_array = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

    participant_clusters = [p1_clust, p2_clust, p3_clust, p4_clust, p5_clust, p6_clust, p7_clust, p8_clust, p9_clust, p10_clust]

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

def visualize(x,y,cluster):
    plt.colorbar(plt.scatter(x,y,c=cluster))
    plt.legend()
    plt.xlabel("Trial Number")
    plt.ylabel("Participant Number")
    plt.show()

if __name__ == "__main__":
    # Gets array of all the individual participants clustered by trial
    ind_participants = run_singles()

    # A Participant containing data for channel 1 across all participants, all trials
    c1_array, c1_labels = get_channel(np.array(ind_participants), 1)
    c1 = p("c1", fv=c1_array, labels=c1_labels)

    # A Participant containing all trials, all channels, all participants. Cluster by trial
    all_array, labels_array = get_all_participants(ind_participants)

    p1 = p("01")
    p1_clust, p1_acc = p1.cluster_3D()

    labels_array.reshape([1, len(ind_participants) * 75])
    all = p("all", fv=all_array, labels=labels_array)
    all.labels = labels_array
    all_clusters, all_accuracy = all.cluster_3D(100)
    trials_array=np.tile(np.arange(1,76),10)
    particp_array=np.repeat(np.arange(1,11),75)
    count1_right=0
    count2_right=0
    count3_right=0
    for i in range(75*14):
        if(all.labels[i]==1 and all_clusters[i]==1):count1_right+=1
        elif(all.labels[i]==2 and all_clusters[i]==2):count2_right+=1
        elif(all.labels[i]==3 and all_clusters[i]==3):count3_right+=1
    #visualize(trials_array, particp_array, all_clusters)
    nums=["1","2","3"]
    count_arr=[count1_right/3.5,count2_right/3.5,count3_right/3.5]
    print(count_arr)
    plt.bar(nums, count_arr, alpha=0.7)
    plt.xlabel("Trial Type")
    plt.ylabel("Percent Right")
    plt.show()

    # plt.scatter(particp_array,all_clusters, c=all.labels)
    # plt.xlabel("Trial Number")
    # plt.ylabel("Percent clustering similarity")
    # plt.show()

    # channels, channel_labels = get_channel(np.array(ind_participants),1)
    # print(channels)
    # channel_indices=np.full(40,1)
    # for i in range (2,40):
    #     channel, label=get_channel(np.array(ind_participants),i)
    #     channels=np.append(channels, channel)
    #     channel_labels=np.append(channel_labels, label)
    #     channel_indices=np.append(channel_indices, np.full(40,i))
    # print(channels)
    # c = p('c', fv=channels, labels=channel_labels)
    # channel_clusters = c.cluster_2D(2)
    # print(channel_indices)
    #plt.scatter(channel_indices, channel_labels, c=channel_clusters)

