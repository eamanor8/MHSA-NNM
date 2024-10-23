import pandas as pd
from io import StringIO

users = '100'

# Read and truncate the dataset to exclude the first 100 unique users
input_file = './data/gowalla/original-Gowalla_totalCheckins.txt'  # Path to the original file
remaining_output_file = './data/gowalla/Gowalla_totalCheckins.txt'  # Path to save the remaining user data
excluded_output_file = './data/gowalla/%s-excluded-users.txt' % users  # Path to save the excluded users' data

# Load the dataset
with open(input_file, 'r') as infile:
    lines = infile.readlines()

# Assuming the format is tab-separated values
df = pd.read_csv(StringIO(''.join(lines)), sep='\t', header=None, names=["user_id", "date-time", "latitude", "longitude", "location_id"])

# Get the unique user IDs
unique_users = df['user_id'].unique()

# Separate the first 100 users
excluded_users = unique_users[:100]

# Filter out the excluded users
remaining_users_data = df[df['user_id'].isin(unique_users[100:])]
excluded_users_data = df[df['user_id'].isin(excluded_users)]

# Save the remaining dataset to a new file
with open(remaining_output_file, 'w') as outfile:
    remaining_users_data.to_csv(outfile, sep='\t', header=False, index=False)

# Save the excluded users' dataset to a new file
with open(excluded_output_file, 'w') as outfile:
    excluded_users_data.to_csv(outfile, sep='\t', header=False, index=False)

print(f'Remaining user data saved to {remaining_output_file}')
print(f'Excluded user data saved to {excluded_output_file}')
