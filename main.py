import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd


'''
Euclidean distance calculation
'''
def euclidean_clustering_epoch(data: np.ndarray, just_return_lowest_distance: bool = False) -> tuple:
    # Initializing the distances matrix to zeros
    distances_matrix = np.zeros((len(data), len(data)))
    # Initializing the minimum values
    min_distance = 1000
    min_data_indexes = []
    # Executing the distances calculation
    for i in range(len(data)):
        j = i
        while j < len(data):
            euclidean_distance = np.around(np.linalg.norm(data[i] - data[j]), 2)
            distances_matrix[i][j] = euclidean_distance

            if euclidean_distance and euclidean_distance < min_distance:
                min_distance = euclidean_distance
                min_data_indexes = [i, j]
                
            j += 1

    if just_return_lowest_distance:
        return (min_distance, min_data_indexes)

    print('Distance matrix:')
    print(distances_matrix)
    print()
    print(f'Shortest distance: {min_distance}. In labels: {min_data_indexes[0]+1} and {min_data_indexes[1]+1}')
    print()

    data[min_data_indexes[0]] = np.divide(np.add(data[min_data_indexes[0]], data[min_data_indexes[1]]), 2.0)
    
    data = np.delete(data, min_data_indexes[1], 0)

    return (data, min_data_indexes)


'''
Dendrogram plotting
'''
def dendrogram_plotting(data, labels, y_limit):
    # Plotting the data in a dendrogram
    dendrogram = sch.dendrogram(sch.linkage(data, method = 'median'), labels=labels)
    plt.axhline(y=y_limit, linewidth=3, color='r') # Drawing a limit to see more clearly the clustering classes
    plt.title('Dendrogram')
    plt.xlabel('Labels')
    plt.ylabel('Euclidean distances')
    plt.show()


'''
Radar Plotting
'''
def radar_plotting(data: np.array, data_labels: list[str], attr_labels: list[str]):
    # Plotting the data in a radar plot
    # Setting the data size for radar plotting
    data = np.append(data, [data[0]], axis=0)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(polar=True)
    # Setting the angles based on the number of data labels
    angles = np.linspace(0, 2*np.pi, len(data_labels), endpoint=False)
    angles = np.concatenate((angles,[angles[0]]))
    # Completing the data labels size
    data_labels.append(data_labels[0])
    # Plotting the data
    ax.plot(angles, data,'o--', label=attr_labels)
    # Changing the axis labels
    ax.set_thetagrids(angles * 180/np.pi, data_labels)
    plt.legend() # To show the legend with attributes labels
    plt.title('Radar Plot')
    plt.show()


'''
Main Function
'''
def main():
    # Initializing the data by reading it from CSV file
    data = pd.read_csv(csv_file_name)
    data_np = data.to_numpy(dtype=np.dtype(float), copy=True)
    # Initializing the data labels
    data_attr_labels = list(data.columns.values)
    data_labels = [str(i+1) for i in range(len(data_np))]
    data_labels_for_plotting = data_labels.copy()

    # Clustering execution
    while len(data_labels) > number_of_clusters:
        print('-'*50)
        data_np, indexes_changed = euclidean_clustering_epoch(data_np)

        data_labels[indexes_changed[0]] = data_labels[indexes_changed[0]] + '-' + data_labels[indexes_changed[1]]
        data_labels.pop(indexes_changed[1])

        print('New clusters:')
        [print("|{:15}|{:20}|".format(data_labels[i], str(data_np[i]))) for i in range(len(data_np))]

        print('-'*50)

    # Plotting section
    min_distance, _ = euclidean_clustering_epoch(data_np, True)
    dendrogram_plotting(data.to_numpy(copy=True), data_labels_for_plotting, min_distance)
    radar_plotting(data.to_numpy(copy=True), data_labels_for_plotting, data_attr_labels)


if __name__ == '__main__':
    # CSV file name
    csv_file_name = input('Enter the name of CSV file: ')

    # Number of final clusters
    number_of_clusters = int(input('Enter the number of final clusters: '))

    try:
        main()
    except Exception as e:
        print(str(e))