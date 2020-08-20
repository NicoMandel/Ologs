#!/usr/bin/env python3

######## TODO:
# 0. Include lexnames: https://wordnet.princeton.edu/documentation/lexnames5wn
# 1. When lists in holonyms etc. are empty, return None instead of empty lists through checks whether isempty or len() > 0
# 2. Look @ closure from the methods of ntlk.wn
# 3. Implement Sussna's Depth-relative scaling as mentioned in: https://dl.acm.org/citation.cfm?id=1168108 - requires 
# using the name of a method as an input argument (maybe coming from one of the visualisation tasks)
# 4. Look @ Domains and Topics as mentioned here: http://spandanagella.com/files/lrec_wn_domains_poster.pdf - implemented on lemmas
# 5. Include a concept similar to IC - information content - into visual / location data - relative importance of occurence!
# 6. Look at the Visual Distance in WordNet Paper: https://arxiv.org/abs/1804.09558
# 7. Using the potential ACTION keyword from Gavin, we can look at other relations than just hypernymy and hyponymy
# 8. USE WordNetLemmatizer !!!!!
#########################

# Abstract class to provide interface to work with the nltk toolkit
# Research (https://dl.acm.org/citation.cfm?id=1168108) shows, that 80% of all WN relations are holonym, hypernym
# and inclusion of other relationships may not be beneficial
# Limitations:
# 1. Functions part_meronym and part_holonym currently have a depth of 1
# 2. Antonyms are defined over lemmas and not synsets

import nltk
from nltk.corpus import wordnet as wn
import math


class SearchNode:
    """ By Gavin Suddrey - Perfect work! 
    A Search Node Class. Stores the state and the the parent upon initializing. Args: State (ergo a synset), Param: Parent (also a synset). Returns None""" 
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

    def path(self):
        """Path to a node. Returns the unrolled path of all parents as SearchNode objects"""
        parent = self.parent
        _path = [self]
        while parent:
            _path.append(parent)
            parent = parent.parent
        return _path
        

class wnclient:
    """ WordNet Client class to work with objects from Gazebo in WN. Empty constructor. Maximum Hyponym count = 39555 (for "Physical Entity")
        Maximum Taxonomy depth in WN is 19. synsets_clean only retains the synsets which have "physical entity" in their hypernym path!
        CAREFUL: The root hypernym count in WN implements "entity" as the root hypernym... But since the count is often log scaled, we do not reimplement
        a correction for this error.

        ALSO CAREFUL: All relationships are defined over synsets - clearing out the data to make sure the correct synset is used
        Use for loops for simple lemmas/words
        ALSO CAREFUL: All semantic evaluations are made on "Similarity" and not on "Relatedness" of stuff - opposite things are related nonetheless
        The 4 Relationships defined are:
        1. Hyponymy
        2. Hypernymy
        3. Part - Meronymy (Not Substance meronymy)
        4. Member - Holonymy

        ________________________

        Currently implemented methods:
        synsets(lemma) = list of synsets for this lemma
        antonyms(lemma) = list of antonyms for said lemma (if exists)
        meronyms(synset OR lemma) = list of part_meronyms (if exists)
        holonyms(synset OR lemma) = list of member_holonyms (if exists)

        ____________________________

        Similarities:
        Path - similarity: Shortest connecting path in hypernym/holonym fashion - 1 indicates similarity
        Leacock - chodrow similarity: Normalisation for parent depth in taxonomy - -log(p/2d)
        Wu-Palmer similarity: Normalisation for own depth and parent depth in Taxonomy
    """
    # Global variables
    MAX_HYPO = 39555 # Value from "Physical Entity"
    MAX_DEPTH = 19
    root_synset = wn.synsets('physical_entity')[0] 

    def __init__(self):
        """ Empty Constructor """
        # self.root_synset = self.synsets('physical_entity')[0]

    @staticmethod
    def _global_maximum_hypos():
        """ Hidden Helper Function to calculate the maximum number of possible hyponyms (To normalize IC). Run only once. Args: None, Returns: None
        ______________________
        Results:
        Structure:
        1. entity = 74373
            1.1 physical entity = 39555
                1.1.1 thing = 2354
                1.1.2 object = 29580
                ...
        1.2 abstraction = 38668
        """
        for synset in list(wn.all_synsets('n'))[:5]:
            hypo_length = 0
            print(synset)
            hypo_length += len(list(synset.closure(hypo)))
            print(hypo_length)

    @staticmethod
    def _maximum_depth():
        """ Helper function. Run only ONCE to calculate the maximum depth of the Taxonomy of the part of speech.
        Based on: wordnetcorpusreader._compute_max_depth - https://gist.github.com/stevenbird/11128654#file-wordnet-py-L1062
        Used in LCH calculation ALL THE FUCKING TIME, that is why it is so slow
        """
        depth = 0
        for syns in wn.all_synsets(wn.NOUN):
            try:
                depth = max(depth, syns.max_depth())
            except RuntimeError:
                print(syns)
        print("The maximum depth of the entire taxonomy is: ", depth)
        return depth

    # Global lambda functions for all 4 relevant relationships
    hypo = classmethod(lambda cls, s: s.hyponyms())
    hyper = staticmethod(lambda s: s.hypernyms())
    meros = staticmethod(lambda s: s.part_meronyms())
    holos = staticmethod(lambda s: s.member_holonyms())
    # Lambda functions to get all synsets for a lemma 
    synsets = staticmethod(lambda lemma: wn.synsets(lemma, pos=wn.NOUN))
    # Lambda functions to get the lowest common hypernyms
    lowest = staticmethod(lambda s_one, s_two: s_one.lowest_common_hypernyms(s_two))
    # Distance lambda functions:
    path_simil = staticmethod(lambda s_one, s_two: s_one.path_similarity(s_two))
    lch_simil = staticmethod(lambda s_one, s_two: s_one.lch_similarity(s_two))
    wup_simil = staticmethod(lambda s_one, s_two: s_one.wup_similarity(s_two))
    num_hypos = classmethod(lambda cls, s: len(list(s.closure(cls.hypo))))


    @staticmethod
    def get_lexmembers(lexname):
        """ Gets the list of all synsets of a lexicon. Please see [here](https://wordnet.princeton.edu/documentation/lexnames5wn) for more information. 
            Raises a TypeError if the lexname is not in the system. Iterates through ALL synsets, may take a while.
            Returns: A list of synsets
        """
        try:
            lexlist = [ss for ss in wn.all_synsets(pos='n') if lexname in ss.lexname()]
            return lexlist
        except:
            raise TypeError("Unknown Lexname. Please try again")

    ################
    ### ROS FUNCTIONS
    ################

    ### Changed methods from lambda here for integration with ROS services
    ### raising errors and converting to synsets first
    @staticmethod
    def hypo_func(concept):
        """ Direct Hyponyms converted to be operable with ROS service
            1. Throwing TypeError
            2. Converting string to synset first
        """
        try:
            syns = wn.synset(concept)
            hypos = syns.hyponyms()
            return hypos
        except:
            raise TypeError("Unkown synset definition")

    # Function for the direct hypernyms
    @staticmethod
    def hyper_func(concept):
        """ Direct Hypernyms converted to be operable with ROS service
            1. Throwing TypeError
            2. Converting string to synset first
        """
        try:
            syns = wn.synset(concept)
            hypers = syns.hypernyms()
            return hypers
        except:
            raise TypeError("Unkown synset definition")
        
    # Function for the path similarity
    @staticmethod
    def path_simil_func(conc_1, conc_2):
        """ Path-Simliarity converted to be operable with ROS service
            1. Throwing TypeError
            2. Converting string to synset first
        """
        try:
            syn_1 = wn.synset(conc_1)
            syn_2 = wn.synset(conc_2)
            dist = syn_1.path_similarity(syn_2)
            return dist 
        except:
            raise TypeError("Unkown synset definition")

    # Function for the leacock-chodrow similarity
    @staticmethod
    def lch_simil_func(conc_1, conc_2):
        """ Leacock-Chodrwo similarity converted to be operable with ROS service
            1. Throwing TypeError
            2. Converting string to synset first
        """
        try:
            syn_1 = wn.synset(conc_1)
            syn_2 = wn.synset(conc_2)
            dist = syn_1.lch_similarity(syn_2)
            return dist 
        except:
            raise TypeError("Unkown synset definition")

    # Function for the wu-palmer similarity
    @staticmethod
    def wup_simil_func(conc_1, conc_2):
        """ Wu-Palmer similarity converted to be operable with ROS service
            1. Throwing TypeError
            2. Converting string to synset first
        """
        try:
            syn_1 = wn.synset(conc_1)
            syn_2 = wn.synset(conc_2)
            dist = syn_1.wup_similarity(syn_2)
            return dist 
        except:
            raise TypeError("Unkown synset definition")

    # Function for the holonyms
    @staticmethod
    def holos_func(concept):
        """ direct Holonyms converted to be operable with ROS service
            1. Throwing TypeError
            2. Converting string to synset first
        """
        try:
            syn = wn.synset(concept)
            hololist = syn.member_holonyms()
            return hololist
        except:
            raise TypeError("Unkown synset definition")

    # Function for the Meronyms
    @staticmethod
    def meros_func(concept):
        """ Direct Meronyms converted to be operable with ROS service
            1. Throwing TypeError
            2. Converting string to synset first
        """
        try:
            syn = wn.synset(concept)
            merolist = syn.part_meronyms()
            return merolist
        except:
            raise TypeError("Unkown synset definition")

    # Function for the lowest common hypernym
    @staticmethod
    def lowest_func(conc_1, conc_2):
        """ Lowest-common-Hypernym converted to be operable with ROS service
            1. Throwing TypeError
            2. Converting string to synset first
        """
        try:
            syn_1 = wn.synset(conc_1)
            syn_2 = wn.synset(conc_2)
            lowest = syn_1.lowest_common_hypernyms(syn_2)[0]
            return lowest
        except:
            raise TypeError("Unkown synset definition")

    # Function to get the connecting path
    @classmethod
    def connecting_path_func(cls, conc_1, conc_2):
        """ Method to call the connecting_path function after some cleaning:
            Method converted to be operable with ROS service
            1. Throwing TypeError
            2. Converting string to synset first
        """
        try:
            syn_one = wn.synset(conc_1)
            syn_two = wn.synset(conc_2)
            path = cls.connecting_path(syn_one, syn_two)
            return path
        except:
            raise TypeError("Unkown synset definition")

    # Function to get the lexicon
    @staticmethod
    def lexname_func(conc_1):
        """ Lexname Method converted to be operable with ROS service
            1. Throwing TypeError
            2. Converting string to synset first
        """
        try:
            syn = wn.synset(conc_1)
            return syn.lexname()
        except:
            raise TypeError("Unkown synset definition")

    # Function to get the definition
    @staticmethod
    def get_definition(conc_1):
        """ Returns the synset definition string """
        try:
            syn = wn.synset(conc_1)
            return syn.definition()
        except:
            raise TypeError("Unkown synset definition")


    ################
    ### CLEANING FUNCTIONS
    ################

    @classmethod
    def synsets_clean(cls, lemma):
        """Gets the synsets and cleans them, ergo: uses the methods inbuilt to the synsets to fill the all_hypernyms. Analogous to: https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html
        This function checks whether the root_hypernym ("Phyiscal entity") is present in the hypernym path. If it is not, the synset will be eliminated
        Args: A Lemma
        Returns: A list of synsets cleaned up by the root_synset
        """
        synsets = wn.synsets(lemma,pos=wn.NOUN)
        cleaned = []
        for synset in synsets:
            if not synset._all_hypernyms:
                synset._all_hypernyms = set(self_synset 
                for self_synsets in synset._iter_hypernym_lists() 
                for self_synset in self_synsets)
            if cls.root_synset in synset._all_hypernyms:
                cleaned.append(synset)
        return cleaned

    ########### SECTION 1 - WORDNET PRELIMINARIES  ################
    @classmethod
    def antonyms(cls, input):
        """ Finding the antonyms to a word or synset. Only for Nouns. Note: Antonyms are defined on the lemmas
        Args: A lemma or a synset
        Returns: List of Antonym-Synsets. Potentially empty"""

        antonyms = []
        if type(input) is str:
            synset = cls.synsets(input)
            for syn in synset:
                for lemma in syn.lemmas():
                    if lemma.antonyms():
                        for anton in lemma.antonyms():
                            antonyms.append(anton)
            return antonyms
        else:
            for lemma in input.lemmas():
                if lemma.antonyms():
                    for anton in lemma.antonyms():
                        antonyms.append(anton)
            return antonyms  

    # Part-Meronyms: A Synsets' Components (branches for a tree) - might be a corpse
    @classmethod
    def meronyms(cls, input):
        """ Returns the components of the specified input.
        Args: Lemma OR Synset, Returns: (list of) Synsets. Potentially empty"""
        meronymlist = []
        if type(input) is str:
            synset = cls.synsets(input)
            for syn in synset:
                if syn.part_meronyms():
                    for meronym in syn.part_meronyms():
                        meronymlist.append(meronym)
            return meronymlist
                
        else:
            if input.part_meronyms():
                return input.part_meronyms()
            else: return meronymlist

    # Holonyms: If the object is a component of something bigger - might be a corpse 
    @classmethod
    def holonyms(cls, input):
        """ Returns the collective terms for a specific thing. E.g. Forest for tree.
        Args: Lemma OR Synset, Returns: (list of) Synsets. Potentially empty"""
        holonymlist = []
        if type(input) is str: 
            synset = cls.synsets(input)
            for syn in synset:
                if syn.member_holonyms():
                    for member in syn.member_holonyms():
                        holonymlist.append(member)
            return holonymlist
        else:
            if input.member_holonyms():
                return input.member_holonyms()
            else: return holonymlist            
    

    ################### 
    # IC Calculation
    ##############
    @classmethod
    def intrinsic_IC(cls, hypocount):
        """ Calculating the intrinsic Information content. Formula IC = (log((hypo(c)+1)/max)/log(1/max)) = 1 - [log(hypo(c)+1)/log(max)]
            Args: Maximum number possible (only calculated once, basically a global), and number of hyponyms. IC is between 0 and 1"""
        numer = math.log10(hypocount+1)
        divisor = math.log10(cls.MAX_HYPO)
        return 1-(numer/divisor)

    ## Similarities with IC
    # Resnik similarity - could be implemented as a lambda, but that would make it unnecessarily complex
    @classmethod
    def resnik_similarity (cls, input_one, input_two):
        """ The Resnik similarity uses the IC of the least common subsumer as the similarity value: IC (LCS).
        Args: two lemmas OR synsets to put into funct: lowest_subsumer, which returns a synset and the distance, plus global maximum number of hypos for IC calculation
        Returns: the IC of the lowest common subsumer
        (Careful. This is not the lowest common subsuming IC, which could be smaller) """
        lowest = cls.lowest(input_one, input_two)[0]
        count = cls.num_hypos(lowest)
        return cls.intrinsic_IC(count)

    # Lin - similarity
    @classmethod
    def lin_similarity (cls, input_one, input_two):
        """ Lin Similarity = 2*resnik similarity/ (IC(c1) + IC(c2))
        Args: Two lemmas OR Synsets and the max_global Hyponym count for calculating the resnik similarity.
        Returns: Lin-similarity """
        hypos_1 = cls.num_hypos(input_one)
        hypos_2 = cls.num_hypos(input_two)
        ic_1 = cls.intrinsic_IC(hypos_1)
        ic_2 = cls.intrinsic_IC(hypos_2)
        resnik = cls.resnik_similarity(input_one, input_two)
        lin = (2*resnik / (ic_1+ic_2))
        return lin

    # JCN - Distance
    @classmethod
    def jcn_distance (cls, input_one, input_two):
        """ Jiang & Conrath - Distance = 1/( IC(c1) + IC(c2) - 2*res(c1,c2)).
        Careful: is a distance measure, not a similarity!
        Args: Two Synsets to compare.
        Returns: Distance"""
        resnik = resnik_similarity(input_one, input_two)
        hypos_1 = cls.num_hypos(input_one)
        hypos_2 = cls.num_hypos(input_two)
        ic_1 = intrinsic_IC(hypos_1)
        ic_2 = intrinsic_IC(hypos_2)
        jcn = 1/(ic_1 + ic_2 - (2*resnik))
        return jcn

    ################
    ### PATH FUNCTIONS
    ################

    # Connecting paths
    @classmethod
    def connecting_path(cls, syn_one, syn_two):
        """ To find the path connecting nodes. Stores all nodes along the path. Breadth-first search from two bottom nodes to their LCS
        Args: Two synsets. Returns: a list of SearchNode objects, which each hold a state and a Parent. Those in Turn are Synsets"""
        subsumer = cls.lowest(syn_one, syn_two)[0] # returns a list of subsumers. The original implementation uses the same shorthand

        node1 = cls.search(syn_one, subsumer)
        node2 = cls.search(syn_two, subsumer)    
        path = list(reversed(node1.path())) + node2.path()
        return path
        
    @classmethod
    def search(cls, initial, goal):
        frontier = [SearchNode(initial)]
        explored = set()
        
        while frontier:
            selected = frontier.pop(0)
            explored.add(selected.state)

            if selected.state == goal:
                return selected

            #for s in cls.hyper(selected.state):
            #    if s not in explored:
            #        frontier.append(SearchNode(state = s, parent=selected))
            frontier.extend(SearchNode(state=s, parent=selected) for s in cls.hyper(selected.state) if s not in explored)

    @classmethod
    def explore(cls, initial, iters=10, fn=meros):
        """ Explores the Meronyms to a certain depth. Param fn can be changed for another valid lambda of this class.
        Returns a list of SearchNode objects."""
        frontier = [SearchNode(initial)]
        explored = set()
        while frontier and iters > 0:
            # take the first element out of that list
            selected = frontier.pop(0)
            explored.add(selected)
            frontier.extend(SearchNode(state=s, parent=selected) for s in fn(selected.state) if SearchNode(state=s, parent=selected) not in explored)
            iters -= 1 
        return explored

            


