#!/usr/bin/env python 
# trial run for the path of the wnclient

############### ATTENTION ################
# The AHP will only work when the values are somewhat minutely inconsistent - NOT if they are fully consistent - then a normalised value 
# will be sufficient enough - see relVals in function distToGoal 

from wnclient import *
import networkx as nx       
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np 

# import AHP
import pandas as pd

# Gazebo imports
from Gazebo_client import GazeboModelClient  


class NodePair:

    def __init__(self, synset, parent):
        self.synset = synset
        self.parent = parent
    
    def __str__(self):
        return str(self.parent, " ", self.synset)

    def __repr__(self):
        return (str(self.parent, " ", self.synset))


def showGraph(synset,iters=10,function=wnclient.meros):
    """ Takes a Synset and shows the Meronym Graph"""
    graph = nx.OrderedDiGraph()# Directed Graph
    path = wnclient.explore(synset,iters=iters,fn=function)
    for node in path:
        graph.add_node(node.state.lemmas()[0].name())
        if node.parent:
            graph.add_edge(node.state.lemmas()[0].name(), node.parent.state.lemmas()[0].name())
        print(node.state.lemmas()[0].name())
    pos = graphviz_layout(graph)
    nx.draw_networkx(graph)
    plt.show()

def showConnectGraph(syn1, syn2):
    """ Takes two synsets and shows the graph connecting both
    Args: Two Synsets
    Returns: None"""

    graph = nx.OrderedDiGraph()# Directed Graph
    path = wnclient.connecting_path(syn1, syn2)
    for node in path:
        graph.add_node(node.state.lemmas()[0].name())
        if node.parent:
            graph.add_edge(node.state.lemmas()[0].name(), node.parent.state.lemmas()[0].name())
    pos = graphviz_layout(graph)
    nx.draw_networkx(graph)
    plt.show()

def showConnectGraphSubsets(syn1, syn2):
    """ Shows the graph connecting two subsets and their part holonyms. Based on function showConnectGraph
    Args: Two Synsets - Potential extension through function - for holonyms or meronyms
    Returns: None
    """
    graph = nx.OrderedDiGraph()
    path = wnclient.connecting_path(syn1, syn2)
    for node in path:
        graph.add_node(node.state.lemmas()[0].name())
        holonymList = wnclient.holos(node.state)
        meronymList = wnclient.meros(node.state)
        if node.parent:
            graph.add_edge(node.state.lemmas()[0].name(), node.parent.state.lemmas()[0].name(),color='r',width=8)
        for holonym in holonymList:
            graph.add_node(holonym.lemmas()[0].name())
            graph.add_edge(holonym.lemmas()[0].name(),node.state.lemmas()[0].name(),color='g')
        for meronym in meronymList:
            graph.add_node(meronym.lemmas()[0].name())
            graph.add_edge(meronym.lemmas()[0].name(),node.state.lemmas()[0].name(),color='b')
    
    edges = graph.edges()
    edge_colors = [graph[u][v]['color'] for u,v in edges]
    nx.draw_networkx(graph, edges=edges, edge_color = edge_colors)
    pos = graphviz_layout(graph)
    plt.show()

def distanceMatrix (nameList):
    """ Uses the Path-similarity to calculate the Distance Matrix of all objects to themselves.
    For multiple Synsets of the same Lemma, it averages over them.
    Arg: a list of names to work with, Param::a goal, defaults to "person"
    Returns: A Pandas dataframe with the original parent Lemma as headers"""
    NodeList = []
    for name in nameList:
        SynsetList = wnclient.synsets(name)
        for synset in SynsetList:
            NodeList.append(NodePair(synset, name))
    # here, Nodelist has all the infos - the synset name and the parent name
    #emptyMatrix = np.ones((len(NodeList), len(NodeList))) * -1
    ParentList = [node.parent for node in NodeList]
    #SynsetList = [node.synset for node in NodeList]
    #dfObj = pd.DataFrame(emptyMatrix,index=ParentList,columns=SynsetList)
    

    uniqueList = set(ParentList)
    matrixSums = np.zeros((len(uniqueList), len(uniqueList)))
    matrixCts = np.zeros((len(uniqueList), len(uniqueList)))
    dfSums = pd.DataFrame(data=matrixSums, index=uniqueList,columns=uniqueList) 
    dfCts = pd.DataFrame(data=matrixCts, index=uniqueList, columns=uniqueList)
    
    for i, Node in enumerate(NodeList):
        for j, Node2 in enumerate(NodeList):
            if Node.parent == Node2.parent:
                value = 1.0
            elif i>j:
                continue
            else:
                # this is the part where the actual distance gets calculated
                value = wnclient.path_simil(Node.synset, Node2.synset)
            dfSums.at[Node.parent, Node2.parent] += value
            dfSums.at[Node2.parent, Node.parent] += value
            # pandas uses rows, columns indexing
            dfCts.at[Node.parent, Node2.parent] += 1
            dfCts.at[Node2.parent, Node.parent] += 1
    
    avgs = dfSums.values/dfCts.values
    dfavg = pd.DataFrame(data=avgs, index=uniqueList, columns=uniqueList)
    print(dfavg)
    print("works")
    return dfavg

    # # 3. Pairwise comparison of the names to get the distances - use path similarity - stable and not as badly implemented as JCN 
    # # Take the parent value here, to average over all distances
    # ParentList = [node.parent for node in NodeList]
    # matrix =  np.ones((len(NodeList), len(NodeList))) * -1
    # uniqueList = set(ParentList)
    # matrixSums = np.zeros((len(uniqueList), len(uniqueList))) 
    # matrixCts = np.zeros((len(uniqueList), len(uniqueList)))
    # idxDict = {k: v for v,k in enumerate(uniqueList)}
    # for i, Node in enumerate(NodeList):
    #     for j, Node2 in enumerate(NodeList):
            
    #         if matrix[i,j] >= 0 and i != j:
    #             continue
    #         if Node.parent == Node2.parent or i>j:
    #             continue
    #         # This is the part where the distance actually gets calculated
    #         value = wnclient.path_simil(Node.synset, Node2.synset)
    #         matrix[i,j] = value
    #         matrix[j,i] = value

    #         matrixSums[idxDict[Node.parent],idxDict[Node2.parent]] += value
    #         matrixSums[idxDict[Node2.parent],idxDict[Node.parent]] += value
    #         matrixCts[idxDict[Node.parent],idxDict[Node2.parent]] += 1
    #         matrixCts[idxDict[Node2.parent],idxDict[Node.parent]] += 1
                  
    # avgMat = matrixSums/matrixCts
    # df = pd.DataFrame(data=avgMat,columns=uniqueList)        
    # return df
    
def distToGoal(nameList, goal="person"):
    """ Calculates the semantic distance from all objects to a Goal using the Path similarity
    Args: A List of Names (lemmas). Param: A Goal (defaults to person.).
    Returns: A Pandas Dataframe, ready to be thrown into an AHP) """

    NodeList = []
    goalSyns = wnclient.synsets(goal)
    for name in nameList:
        SynsetList = wnclient.synsets(name)
        for synset in SynsetList:
            NodeList.append(NodePair(synset, name))
    ParentList = set([node.parent for node in NodeList])
    array = np.zeros((len(ParentList),len(goalSyns)))
    arrayCts = np.zeros((len(ParentList),len(goalSyns)))    # Dual entries in case of referencing
    dfSums = pd.DataFrame(array,index=ParentList,columns=goalSyns)
    dfCts = pd.DataFrame(arrayCts,index=ParentList,columns=goalSyns)

    for target in goalSyns:
        for Node in NodeList:
            value = wnclient.path_simil(Node.synset, target)
            dfSums.at[Node.parent,target] += value
            dfCts.at[Node.parent,target] += 1
     
    avgs = dfSums.values/dfCts.values
    dfAvg = pd.DataFrame(data=avgs, index=ParentList, columns=goalSyns)
    
    # Calculating the average over multiple instances of the goal
    DistAvg = {}
    vals = []
    # Calculates the average distance for each parent to the goal
    for parent in ParentList:
        vec =dfAvg.loc[parent]
        avg = np.mean(vec)
        DistAvg[parent] = avg
        vals.append(avg)
    # calculates the average of the average distances
    avg =np.mean(vals)
    vals = vals /avg

    ###########
    # CAREFUL : These three lines do what the AHP does - since the values stem from a mean vector, the consistency is 0 and therefore the 
    # Eigenvector will return to these values - 
    # This could be omitted by using the different goal values to calculate the ratios
    ##########
    print(vals)
    valSum = np.sum(vals)
    relVals = vals/valSum
    print(relVals)

    
    # turns the ratio over the average of the average distance into a dictionary
    for i, parent in enumerate(ParentList):
        DistAvg[parent] = vals[i]
    # creates a dataframe, which calculates a ratio between the distance of one object to the other normalised by the mean 
    # for safety, this does not take the inverse values across the diagonal, but calculates them from scratch
    finaldata = np.ones((len(ParentList),len(ParentList)))
    finaldf = pd.DataFrame(data=finaldata,index=ParentList,columns=ParentList)
    for parent in ParentList:
        for child in ParentList:
            # rows, columns
            finaldf.at[parent,child] = DistAvg[parent] / DistAvg[child]
    return finaldf

 # Currently not functioning completely   

def meronymLevel(name, goal="person"):
    """  # DOES NOT WORK ATM 
    Finds the Part_meronyms of a single lemma to a given depth (preset in wnclient.explore method)
    Arg: A name to search for meronyms 
    Returns: A list of Meronyms
    # CAREFUL:
    Might have to deal with valueoutofrange or indexoutofrange errors here - return None?"""

    MeroList = wnclient.explore(name, fn=wnclient.meros) # A List of SearcNodes
    goalList = wnclient.synsets(goal)
    distArray  = np.array((len(MeroList), len(goalList))) # rows, cols
    merodf = pd.DataFrame(data=distArray,index=MeroList,columns=goalList)
    for meronym in MeroList:
        for target in goalList:
            merodf.at[meronym, target] = wnclient.path_simil(meronym.state,target)

    ########## False calculation, see above!


if __name__=="__main__":
    
    # Normal order of things: 
    # 1. Get the cleaned list of NAMES
    gzclient = GazeboModelClient()
    nameList = gzclient.cleanList
    # removing entries named "clone" - caused by Gazebo Population
    nameListcleaned = []
    for name in nameList:
        if name != 'clone':
            nameListcleaned.append(name)

    # objectList = ["oak_tree", "pine tree", "pine-tree 1", "oak tree5", "building_1"]
    # nameList = ['oak_tree', 'building', 'pine_tree']

    Wncl = wnclient()
    synsetDict = {}
    for name in nameListcleaned:
        synsetDict[name] = wnclient.synsets_clean(name)

    # Showing all possible combinations of the synsets
    for i,(key,values) in enumerate(synsetDict.items()):
        for j, (k,vv) in enumerate(synsetDict.items()):
            if k == key or i>j:
                continue
            for value in values:
                for v in vv:
                    print("Comparing {} with {}".format(value, v))
                    showConnectGraphSubsets(value,v)
    
    print("Done With the Trial!")

    # showing holonyms of all values in the dictionary
    for key, values in synsetDict:
        print(key)
        for value in value:
            showGraph(value,iters=50, function=wnclient.holos)
    print("Test Done")

  
    # Throw stuff into AHP - unneccessary
    df = distToGoal(nameList)
    hierarchy = AHP.analyic_hierarchy(df.values)
    eigenDict = dict(zip(df.keys(),hierarchy.eigVec))
    for k,v in eigenDict.items():
        print (k, v)
    
    # This will get the meronyms for each name and put the pd.Dataframe object into a dictionary
    MeroDict = {}
    for name in nameList:
        MeroDict[name] = meronymLevel(name)
    print("Test Done")




    # print("Test run")
    # oak = wnclient.synsets(nameList[0])[0]
    # print(oak)
    # building = wnclient.synsets(nameList[1])[0]
    # print(building)
    # pine = wnclient.synsets(nameList[2])[0]
    # print(pine)
    # print("Distanace between oak and pine", wnclient.path_simil(pine, oak))
    # print ("Distance between oak and building", wnclient.path_simil(oak, building))
    # print("Distance between pine and building", wnclient.path_simil(pine, building))  




    #except Exception as e:
    #    print(str(e))

