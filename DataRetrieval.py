from Participant import Participant as p
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio

# Run each participant as a separate set, getting the clustering accuracy for each and comparing it to the LOOCV accuracy
def run_singles():
    # LOOCV: 0.826667, 0.533333, 0.573333, 0.960000, 0.53333, 0.786667, 0.760000, 0.746667, 0.666667, 0.853333
    # Participant: 1      2         3          4        5         6         7         8         9        10
    accuracies = []
    rands = []

    p1 = p("01")
    p1_clust, p1_acc, p1_rand = p1.cluster()
    accuracies.append(p1_acc)
    rands.append(p1_rand)

    p2 = p("02")
    p2_clust, p2_acc, p2_rand = p2.cluster()
    accuracies.append(p2_acc)
    rands.append(p2_rand)

    p3 = p("03")
    p3_clust, p3_acc, p3_rand = p3.cluster()
    accuracies.append(p3_acc)
    rands.append(p3_rand)

    p4 = p("04")
    p4_clust, p4_acc, p4_rand = p4.cluster()
    accuracies.append(p4_acc)
    rands.append(p4_rand)

    p5 = p("05")
    p5_clust, p5_acc, p5_rand = p5.cluster()
    accuracies.append(p5_acc)
    rands.append(p5_rand)

    # p6 = p("06")
    # p6_clust, p6_acc = p6.cluster()
    # accuracies.append(p6_acc)

    # p7 = p("07")
    # p7_clust, p7_acc = p7.cluster()
    # accuracies.append(p7_acc)

    # p8 = p("08")
    # p8_clust, p8_acc = p8.cluster()
    # accuracies.append(p8_acc)

    # p9 = p("09")
    # p9_clust, p9_acc = p9.cluster()
    # accuracies.append(p9_acc)

    # p10 = p("10")
    # p10_clust, p10_acc = p10.cluster()
    # accuracies.append(p10_acc)

    # p11 = p("11")
    # p11_clust, p11_acc = p11.cluster()
    # accuracies.append(p11_acc)

    # p12 = p("12")
    # p12_clust, p12_acc = p12.cluster()
    # accuracies.append(p12_acc)

    # p13 = p("13")
    # p13_clust, p13_acc = p13.cluster()
    # accuracies.append(p13_acc)

    # p14 = p("14")
    # p14_clust, p14_acc = p14.cluster()
    # accuracies.append(p14_acc)

    # p15 = p("15")
    # p15_clust, p15_acc = p15.cluster()
    # accuracies.append(p15_acc)

    # p16 = p("16")
    # p16_clust, p16_acc = p16.cluster()
    # accuracies.append(p16_acc)

    # p17 = p("17")
    # p17_clust, p17_acc = p17.cluster()
    # accuracies.append(p17_acc)

    # p18 = p("18")
    # p18_clust, p18_acc = p18.cluster()
    # accuracies.append(p18_acc)

    # p19 = p("19")
    # p19_clust, p19_acc = p19.cluster()
    # accuracies.append(p19_acc)

    # p20 = p("20")
    # p20_clust, p20_acc = p20.cluster()
    # accuracies.append(p20_acc)

    # p21 = p("21")
    # p21_clust, p21_acc = p21.cluster()
    # accuracies.append(p21_acc)

    # p22 = p("22")
    # p14_clust, p22_acc = p22.cluster()
    # accuracies.append(p22_acc)

    # p23 = p("23")
    # p23_clust, p23_acc = p23.cluster()
    # accuracies.append(p23_acc)

    # p24 = p("24")
    # p24_clust, p24_acc = p24.cluster()
    # accuracies.append(p24_acc)

    # p25 = p("25")
    # p25_clust, p25_acc = p25.cluster()
    # accuracies.append(p25_acc)

    # p26 = p("26")
    # p26_clust, p26_acc = p26.cluster()
    # accuracies.append(p26_acc)

    # p27 = p("27")
    # p27_clust, p27_acc = p27.cluster()
    # accuracies.append(p27_acc)

    # p28 = p("28")
    # p28_clust, p28_acc = p28.cluster()
    # accuracies.append(p28_acc)

    # p29 = p("29")
    # p29_clust, p29_acc = p29.cluster()
    # accuracies.append(p29_acc)

    # p30 = p("30")
    # p30_clust, p30_acc = p30.cluster()
    # accuracies.append(p30_acc)

    # participant_array = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30]
    participant_array = [p1, p2, p3, p4, p5]
    print(f"Accuracies for the {len(participant_array)} participants: {accuracies}")
    print(f"Rand Indexes for the {len(participant_array)} participants: {rands}")

    return participant_array, accuracies, rands

def run_channels(individuals):
    channels_array = []
    accuracies = []

    c1_array, c1_labels = get_channel(individuals, 1)
    print("C1 array: ", c1_array)
    print("C1 labels: ", c1_labels)
    c1 = p("c1", fv=c1_array, labels=c1_labels)
    c1_clust, c1_acc = c1.cluster()
    channels_array.append(c1)
    accuracies.append(c1_acc)

    c2_array, c2_labels = get_channel(individuals, 2)
    c2 = p("c2", fv=c2_array, labels=c2_labels)
    c2_clust, c2_acc = c2.cluster()
    channels_array.append(c2)
    accuracies.append(c2_acc)

    c3_array, c3_labels = get_channel(individuals, 3)
    c3 = p("c3", fv=c3_array, labels=c3_labels)
    c3_clust, c3_acc = c3.cluster()
    channels_array.append(c3)
    accuracies.append(c3_acc)

    c4_array, c4_labels = get_channel(individuals, 4)
    c4 = p("c4", fv=c4_array, labels=c4_labels)
    c4_clust, c4_acc = c4.cluster()
    channels_array.append(c4)
    accuracies.append(c4_acc)

    c5_array, c5_labels = get_channel(individuals, 5)
    c5 = p("c5", fv=c5_array, labels=c5_labels)
    c5_clust, c5_acc = c5.cluster()
    channels_array.append(c5)
    accuracies.append(c5_acc)

    c6_array, c6_labels = get_channel(individuals, 6)
    c6 = p("c6", fv=c6_array, labels=c6_labels)
    c6_clust, c6_acc = c6.cluster()
    channels_array.append(c6)
    accuracies.append(c6_acc)

    c7_array, c7_labels = get_channel(individuals, 7)
    c7 = p("c7", fv=c7_array, labels=c7_labels)
    c7_clust, c7_acc = c7.cluster()
    channels_array.append(c7)
    accuracies.append(c7_acc)

    c8_array, c8_labels = get_channel(individuals, 8)
    c8 = p("c8", fv=c8_array, labels=c8_labels)
    c8_clust, c8_acc = c8.cluster()
    channels_array.append(c8)
    accuracies.append(c8_acc)

    c9_array, c9_labels = get_channel(individuals, 9)
    c9 = p("c9", fv=c9_array, labels=c9_labels)
    c9_clust, c9_acc = c9.cluster()
    channels_array.append(c9)
    accuracies.append(c9_acc)

    c10_array, c10_labels = get_channel(individuals, 10)
    c10 = p("c10", fv=c10_array, labels=c10_labels)
    c10_clust, c10_acc = c10.cluster()
    channels_array.append(c10)
    accuracies.append(c10_acc)

    c11_array, c11_labels = get_channel(individuals, 11)
    c11 = p("c11", fv=c11_array, labels=c11_labels)
    c11_clust, c11_acc = c11.cluster()
    channels_array.append(c11)
    accuracies.append(c11_acc)

    c12_array, c12_labels = get_channel(individuals, 12)
    c12 = p("c12", fv=c12_array, labels=c12_labels)
    c12_clust, c12_acc = c12.cluster()
    channels_array.append(c12)
    accuracies.append(c12_acc)

    c13_array, c13_labels = get_channel(individuals, 13)
    c13 = p("c13", fv=c13_array, labels=c13_labels)
    c13_clust, c13_acc = c13.cluster()
    channels_array.append(c13)
    accuracies.append(c13_acc)

    c14_array, c14_labels = get_channel(individuals, 14)
    c14 = p("c14", fv=c14_array, labels=c14_labels)
    c14_clust, c14_acc = c14.cluster()
    channels_array.append(c14)
    accuracies.append(c14_acc)

    c15_array, c15_labels = get_channel(individuals, 15)
    c15 = p("c15", fv=c15_array, labels=c15_labels)
    c15_clust, c15_acc = c15.cluster()
    channels_array.append(c15)
    accuracies.append(c15_acc)

    c16_array, c16_labels = get_channel(individuals, 16)
    c16 = p("c16", fv=c16_array, labels=c16_labels)
    c16_clust, c16_acc = c16.cluster()
    channels_array.append(c16)
    accuracies.append(c16_acc)

    c17_array, c17_labels = get_channel(individuals, 17)
    c17 = p("c17", fv=c17_array, labels=c17_labels)
    c17_clust, c17_acc = c17.cluster()
    channels_array.append(c17)
    accuracies.append(c17_acc)

    c18_array, c18_labels = get_channel(individuals, 18)
    c18 = p("c18", fv=c18_array, labels=c18_labels)
    c18_clust, c18_acc = c18.cluster()
    channels_array.append(c18)
    accuracies.append(c18_acc)

    c19_array, c19_labels = get_channel(individuals, 19)
    c19 = p("c19", fv=c19_array, labels=c19_labels)
    c19_clust, c19_acc = c19.cluster()
    channels_array.append(c19)
    accuracies.append(c19_acc)

    c20_array, c20_labels = get_channel(individuals, 20)
    c20 = p("c20", fv=c20_array, labels=c20_labels)
    c20_clust, c20_acc = c20.cluster()
    channels_array.append(c20)
    accuracies.append(c20_acc)

    return channels_array, accuracies

# Get the ground truth labels from the MATLAB file
def get_channel_names():
    # Load the labels
    mnt = sio.loadmat(f'Data/mnt.mat')
    # print(mnt['mnt']['clab'])

    # Convert from MATLAB struct to np array
    channel_names = []
    for elem in mnt['mnt']['clab'][0][0]:
        for channel in elem:
            channel_names.append(channel[0])

    print("C Names: ", channel_names)
    return channel_names

# Input: participants is an array of all the participants loaded
# Input: channel is an integer between 1-40 indicating which channel to pull
def get_channel(participants, channel):
    channel_array = []
    labels_array = []

    for p in participants:
        for trial in range(0, len(p.fv)):
            # Account for Python being 0-indexed
            channel_array.append([p.fv[trial, channel - 1, :]])
        for elem in p.labels:
            labels_array.append(elem)

    return np.array(channel_array), np.array(labels_array)

# Combines data across all participants
def get_all_participants(participants):
    p_array = []
    labels_array = []

    for p in participants:
        for trial in range(0, len(p.fv)):
            p_array.append(p.fv[trial, :, :])
        for elem in p.labels:
            labels_array.append(elem)

    return np.array(p_array), np.array(labels_array)