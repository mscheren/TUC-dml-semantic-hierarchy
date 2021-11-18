import os
import numpy as np
import networkx as nx

# function for creating the class embeddings
# file structure is assumed to be as follows: every class corresponds to a folder, every folder
# is labeled with a class tag that consists of all the nodes in the tree on the path to the final
# class demarcated by the delimiter '.'
# example: 'bottle.pet.cola' corresponds to all -> bottle -> pet -> cola
# the node 'all' is added as a common origin point

def create_class_embeddings(data_path):
    
    # Create graph

    classes_Graph = list()
    for root, dirs, files in os.walk(data_path):
        for name in dirs:
          classes_Graph.append(name)

    Graph = nx.DiGraph()
    Graph.add_node('all')

    for entry in classes_Graph:
        list_of_entries = entry.split('.')
        length = len(list_of_entries)
        Graph.add_node(list_of_entries[0])
        Graph.add_edge('all',list_of_entries[0])
        for counter in range(length-1):
            list_of_entries[counter+1]='.'.join([list_of_entries[counter],list_of_entries[counter+1]])
            Graph.add_node(list_of_entries[counter+1])
            Graph.add_edge(list_of_entries[counter],list_of_entries[counter+1])
    
    # Create distance matrix (may take a few minutes)

    number_classes = len(classes_Graph)
    distances_matrix = np.zeros((number_classes,number_classes))
    height = nx.dag_longest_path_length(Graph)

    for row in range(number_classes):
        for column in range(number_classes):
            a = classes_Graph[row]
            b = classes_Graph[column]
            if a == b:
                distance = 0
            else:
                source = nx.lowest_common_ancestor(Graph,a,b)
                bunch = list(nx.descendants(Graph,source))
                bunch.append(source)
                sGraph = nx.subgraph(Graph,bunch)
                sheight = nx.dag_longest_path_length(sGraph)
                distance = sheight / height
            distances_matrix[row,column] = distance
            
    # Compute class embeddings

    n = number_classes
    class_embeddings = np.zeros((n,n))
    class_embeddings[0,0] = 1

    for i in range(1,n):
        distance_vector = np.zeros((i,1))
        matrix_prev_embeddings = class_embeddings[:i,:]
        for j in range(i):
            s = 1 - distances_matrix[j,i]
            distance_vector[j] = s
        current_embedding = np.linalg.lstsq(matrix_prev_embeddings,distance_vector)[0]
        class_embeddings[i,:] = current_embedding.T
        class_embeddings[i,i] = np.sqrt(1-np.linalg.norm(class_embeddings[i,:i])**2)
        
    return class_embeddings