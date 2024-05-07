# CSCI4022FinalProject

## Description
Clustering of open-access fNIRS dataset using k-means algorithm

## Running the Algorithm
home.py contains the automated script to run clustering-by-trial for each individual participant, clustering-by-trial for channel 1, and clustering-by-trial for all participants. Information is printed to the terminal and any plots generated are saved to Visualizations/.

To use new data, add the MATLAB struct to the Data/MATLAB folder. Then, run our modified tutorial_classification.m script with version 2020a to save the feature vector and labels to Data/Clustering (be sure to change the variable "dataset" on line 26, under Section 2). home.py will automatically convert the feature vector to the format required for our Python scripting.

## Structure
Data: Contains original MATLAB structs from the paper (in Data/MATLAB), the feature vector and ground-truth labels extracted from the given classification script (in Data/Clustering), and the reorganized data in numpy array format (in Data/NP)

Papers: Contains relevant papers to this project

Visualizations: Contains figures produced by the team

## Credits
Anna Rahn: Converted data from MATLAB structs to numpy arrays, rearranged data for clustering by different combinations of participants, wrote MLP for comparison, designed data visualizations, literature review, final report

Kieran Stone: Wrote k-means algorithm, designed data visualizations, final report

Brodie Schmidt: Literature review, final report
