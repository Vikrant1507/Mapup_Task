from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    length = len(lst)

    for i in range(0, length, n):
        start = i
        end = min(i + n - 1, length - 1)  # Ensure we don't go out of bounds

        # Reverse the current group by swapping elements
        while start < end:
            lst[start], lst[end] = lst[end], lst[start]
            start += 1
            end -= 1

    return lst


# SC-O(N)
# TC-O(1)


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {}

    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        # Append the string to the list for its length
        length_dict[length].append(string)

    # Sort the dictionary by keys and return it
    return dict(sorted(length_dict.items()))


def flatten_dict(nested_dict: Dict, sep: str = ".") -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.

    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    flat_dict = {}

    def _flatten(current_dict: Dict[str, Any], parent_key: str):
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                _flatten(value, new_key)
            elif isinstance(value, list):
                # Recur for lists
                for index, item in enumerate(value):
                    item_key = f"{new_key}[{index}]"
                    if isinstance(item, dict):
                        _flatten(item, item_key)
                    else:
                        flat_dict[item_key] = item
            else:
                flat_dict[new_key] = value

    _flatten(nested_dict, "")
    return flat_dict


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.

    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """

    # Your code here
    def backtrack(start: int):
        if start == len(nums):
            permutations.append(nums[:])  # Append a copy of the current permutation
            return

        seen = set()  # To track duplicates in this position
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue  # Skip duplicates
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]  # Swap
            backtrack(start + 1)  # Recurse
            nums[start], nums[i] = nums[i], nums[start]  # Backtrack

    nums.sort()  # Sort to ensure duplicates are adjacent
    permutations = []
    backtrack(0)  # Start the backtracking process
    return permutations


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.

    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r"(\d{2})-(\d{2})-(\d{4})",  # dd-mm-yyyy
        r"(\d{2})/(\d{2})/(\d{4})",  # mm/dd/yyyy
        r"(\d{4})\.(\d{2})\.(\d{2})",  # yyyy.mm.dd
    ]

    combined_pattern = "|".join(date_patterns)
    matches = re.findall(combined_pattern, text)

    valid_dates = []
    for match in matches:

        if len(match) == 3:  # Assumes all matches are in dd-mm-yyyy format
            valid_dates.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif len(match) == 2:  # assumption for mm/dd/yyyy
            valid_dates.append(f"{match[0]}/{match[1]}/2000")  # Defaulting year to 2000
        elif len(match) == 4:  # structure for yyyy.mm.dd
            valid_dates.append(f"{match[0]}.{match[1]}.{match[2]}")

    return valid_dates


import polyline  # type: ignore


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.

    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)

    df = pd.DataFrame(coordinates, columns=["lat", "lon"])
    df["distance"] = 0

    for i in range(1, len(df)):
        df.loc[i, "distance"] = (df.loc[i, "lat"] - df.loc[i - 1, "lat"]) ** 2 + (
            df.loc[i, "lon"] - df.loc[i - 1, "lon"]
        ) ** 2

    df["distance"] = df["distance"].apply(lambda x: x**0.5)

    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element
    by the sum of its original row and column index before rotation.

    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.

    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]

    return final_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    df["start"] = pd.to_datetime(df["startDay"] + " " + df["startTime"])
    df["end"] = pd.to_datetime(df["endDay"] + " " + df["endTime"])

    grouped = df.groupby(["id", "id_2"])

    def check_completeness(group):
        days = group["start"].dt.dayofweek.unique()
        times = pd.Series([group["start"].min().time(), group["end"].max().time()])
        return (
            len(days) >= 5
            and times.min() == pd.Timestamp("00:00:00").time()
            and times.max() == pd.Timestamp("23:59:59").time()
        )

    result = grouped.apply(check_completeness)

    return result
