from scipy.stats import entropy
def calculate_entropy(probabilities):
    return entropy(probabilities, base=2)


def conditional_entropy(y, x):
    # Calculate H(Y|X)
    return calculate_entropy(y) - calculate_entropy(x)


def maximize_entropy_subset(y, n):
    # Ensure n is within the range of the data
    n = min(n, len(y))

    # Start with an empty subset
    subset = []
    selectedIndex = []

    for _ in range(n):
        max_information_gain = 0
        best_measurement = None
        best_index = None
        for j, measurement in enumerate(y):
            if measurement not in subset and j not in selectedIndex:
                candidate_subset = subset + [measurement]
                info_gain = conditional_entropy(y, candidate_subset)

                if info_gain > max_information_gain:
                    max_information_gain = info_gain
                    best_measurement = measurement
                    best_index = j
        # Add the best measurement to the subset
        subset.append(best_measurement)
        selectedIndex.append(best_index)
    return selectedIndex
