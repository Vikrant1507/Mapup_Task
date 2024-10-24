import pandas as pd
import datetime as dt


def calculate_distance_matrix(df) -> pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    all_ids = pd.Index(df["id_start"].append(df["id_end"]).unique())

    # Create a square matrix filled with infinity (to represent unknown distances)
    distance_matrix = pd.DataFrame(np.inf, index=all_ids, columns=all_ids)

    np.fill_diagonal(distance_matrix.values, 0)

    for _, row in df.iterrows():
        id_start, id_end, distance = row["id_start"], row["id_end"], row["distance"]
        distance_matrix.loc[id_start, id_end] = distance
        distance_matrix.loc[id_end, id_start] = distance  # Symmetric entry

    # Apply Floyd-Warshall Algorithm to find the shortest path between all pairs
    # This algorithm helps in computing cumulative distances for indirectly connected points
    n = len(all_ids)
    for k in all_ids:
        for i in all_ids:
            for j in all_ids:
                if (
                    distance_matrix.loc[i, j]
                    > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]
                ):
                    distance_matrix.loc[i, j] = (
                        distance_matrix.loc[i, k] + distance_matrix.loc[k, j]
                    )

    return distance_matrix


# Example usage
df = pd.read_csv(
    r"C:\Users\samee\Downloads\MapUp-DA-Assessment-2024-main\MapUp-DA-Assessment-2024-main\datasets\dataset-2.csv"
)  # Load your dataset
distance_matrix = calculate_distance_matrix(df)
print(distance_matrix)


def unroll_distance_matrix(df) -> pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    unrolled_data = []
    for i in df.index:
        for j in df.columns:
            if i == j:
                continue
            unrolled_data.append((i, j, df.loc[i, j]))

    unrolled_df = pd.DataFrame(
        unrolled_data, columns=["id_start", "id_end", "distance"]
    )
    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    # Step 1: Calculate the average distance for the reference ID
    ref_df = df[df["id_start"] == reference_id]
    ref_avg_distance = ref_df["distance"].mean()

    # Step 2: Calculate 10% threshold
    threshold = ref_avg_distance * 0.1

    # Step 3: Use floor and ceil to set the lower and upper bounds
    lower_bound = math.floor(ref_avg_distance - threshold)
    upper_bound = math.ceil(ref_avg_distance + threshold)

    # Step 4: Group by id_start and calculate the average distance for each group
    avg_distances = df.groupby("id_start")["distance"].mean().reset_index()

    # Step 5: Filter IDs whose average distance falls within the lwoer and upper bound
    filtered_ids = avg_distances[
        (avg_distances["distance"] >= lower_bound)
        & (avg_distances["distance"] <= upper_bound)
    ]

    # Step 6: Return the filtered DataFrame sorted by id_start
    return filtered_ids.sort_values("id_start")


# Assuming you have a DataFrame df with columns 'id_start', 'id_end', 'distance'
reference_id = 1001400
result = find_ids_within_ten_percentage_threshold(df, reference_id)

# Display the result
print(result)


def calculate_toll_rate(df) -> pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    df["moto"] = df["distance"] * 0.8
    df["car"] = df["distance"] * 1.2
    df["rv"] = df["distance"] * 1.5
    df["bus"] = df["distance"] * 2.2
    df["truck"] = df["distance"] * 3.6

    return df


# Assuming df contains the 'distance' column from Question 10
result = calculate_toll_rate(df)
print(result)


def calculate_time_based_toll_rates(df) -> pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekends = ["Saturday", "Sunday"]

    time_intervals = [
        (dt.time(0, 0), dt.time(9, 0), 0.8),
        (dt.time(9, 0), dt.time(17, 0), 1.2),
        (dt.time(17, 0), dt.time(23, 59, 59), 0.8),
    ]
    weekend_discount = 0.75

    expanded_rows = []

    for _, row in df.iterrows():
        base_moto = row["moto"]
        base_car = row["car"]
        base_rv = row["rv"]
        base_bus = row["bus"]
        base_truck = row["truck"]

        for day in weekdays + weekends:
            if day in weekdays:
                for start_time, end_time, factor in time_intervals:
                    expanded_rows.append(
                        {
                            "id_start": row["id_start"],
                            "id_end": row["id_end"],
                            "start_day": day,
                            "end_day": day,
                            "start_time": start_time,
                            "end_time": end_time,
                            "moto": base_moto * factor + 0.01,
                            "car": base_car * factor + 0.02,
                            "rv": base_rv * factor + 0.03,
                            "bus": base_bus * factor - 0.02,
                            "truck": base_truck * factor,
                        }
                    )
            else:
                expanded_rows.append(
                    {
                        "id_start": row["id_start"],
                        "id_end": row["id_end"],
                        "start_day": day,
                        "end_day": day,
                        "start_time": dt.time(0, 0),
                        "end_time": dt.time(23, 59, 59),
                        "moto": base_moto * weekend_discount - 0.01,
                        "car": base_car * weekend_discount - 0.02,
                        "rv": base_rv * weekend_discount - 0.03,
                        "bus": base_bus * weekend_discount + 0.01,
                        "truck": base_truck * weekend_discount,
                    }
                )

    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df


# Example usage
result = calculate_time_based_toll_rates(df)
print(result)
