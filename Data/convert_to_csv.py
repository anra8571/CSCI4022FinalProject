import scipy.io
import pandas as pd
import numpy as np

# Load the .mat file
mat_filepath = 'fNIRS 04.mat'
mat_contents = scipy.io.loadmat(mat_filepath)

# Assuming 'dat' is a complex structure containing multiple channels or similar
data = mat_contents

# Placeholder for DataFrame columns and data
columns = []
data_for_df = []

# Example process for a hypothetical structured array with multiple channels
# This needs adjustment based on the actual structure of 'dat'
try:
    # Iterate through the structure if it's multi-dimensional or has multiple fields
    for i in range(data.shape[0]):  # Assuming the first dimension is the one we iterate over
        row_data = []
        for j in range(data.shape[1]):  # Iterate over the second dimension if applicable
            if i == 0:  # For the first row, capture column names
                columns.append(f"Channel_{j+1}")
            row_data.append(data[i, j])  # Collect data for the current row
        data_for_df.append(row_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data_for_df, columns=columns)
    
    # Export DataFrame to .csv
    csv_filepath = 'fNIRS_04_converted.csv'
    df.to_csv(csv_filepath, index=False)
    success = True
except Exception as e:
    success = False
    error_message = str(e)

# Output the result
csv_filepath if success else error_message
