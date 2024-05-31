import pandas as pd

def weighted_average(df1, df2, weights):
    """Calculate weighted average for given dataframes and weights."""
    result = pd.DataFrame()
    result['id'] = df1['id']
    for col in weights.keys():
        result[col] = df1[col] * weights[col][0] + df2[col] * weights[col][1]
    return result

def main(file1, file2, output_file, weights):
    # Load the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check if both dataframes have the same IDs in the same order
    if not (df1['id'].equals(df2['id'])):
        raise ValueError("The IDs in both files do not match or are not in the same order.")

    # Calculate weighted averages
    result = weighted_average(df1, df2, weights)

    # Save the result to a new CSV file
    result.to_csv(output_file, index=False)
    print(f"Weighted averages saved to {output_file}")

if __name__ == "__main__":
    # Define file paths
    file1 = '/root/submission_mj.csv'
    file2 = '/root/submission_ky.csv'
    output_file = '/root/submission_final.csv'

    # Define weights for each column (each key is a column, and values are the weights for df1 and df2 respectively)
    weights = {
        'X4': [0.5, 0.5],
        'X11': [0.6, 0.4],
        'X18': [0.7, 0.3],
        'X50': [0.5, 0.5],
        'X26': [0.4, 0.6],
        'X3112': [0.3, 0.7]
    }

    # Run the main function
    main(file1, file2, output_file, weights)