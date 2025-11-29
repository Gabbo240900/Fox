/* 
############################################################################
#                                Coala                                     #
#         CO-evolution Assessment by a Likelihood-free Approach            #
############################################################################
#                                                                          #
# Copyright INRIA 2013                                                     #
#                                                                          #
# Contributors : Christian Baudet                                          #
#                Beatrice Donati                                           #
#                Blerina Sinaimeri                                         #
#                Pierluigi Crescenzi                                       #
#                Christian Gautier                                         #
#                Catherine Matias                                          #
#                Marie-France Sagot                                        #
#                                                                          #
# christian.baudet@inria.fr                                                #
# marie-france.sagot@inria.fr                                              #
# https://gforge.inria.fr/forum/forum.php?forum_id=11324                   #
#                                                                          #
# This software is a computer program whose purpose is to estimate         #
# the probability of events (co-speciation, duplication, host-switch and   #
# loss) for a pair of host and parasite trees through a Approximate        #
# Bayesian Computation - Sequential Monte Carlo (ABC-SMC) procedure.       #
#                                                                          #
# This software is governed by the CeCILL  license under French law and    #
# abiding by the rules of distribution of free software.  You can  use,    # 
# modify and/ or redistribute the software under the terms of the CeCILL   #
# license as circulated by CEA, CNRS and INRIA at the following URL        #
# "http://www.cecill.info".                                                # 
#                                                                          #
# As a counterpart to the access to the source code and  rights to copy,   #
# modify and redistribute granted by the license, users are provided only  #
# with a limited warranty  and the software's author,  the holder of the   #
# economic rights,  and the successive licensors  have only  limited       #
# liability.                                                               #
#                                                                          #
# In this respect, the user's attention is drawn to the risks associated   #
# with loading,  using,  modifying and/or developing or reproducing the    #
# software by the user in light of its specific status of free software,   #
# that may mean  that it is complicated to manipulate,  and  that  also    #
# therefore means  that it is reserved for developers  and  experienced    #
# professionals having in-depth computer knowledge. Users are therefore    #
# encouraged to load and test the software's suitability as regards their  #
# requirements in conditions enabling the security of their systems and/or #
# data to be ensured and,  more generally, to use and operate it in the    # 
# same conditions as regards security.                                     #
#                                                                          #
# The fact that you are presently reading this means that you have had     #
# knowledge of the CeCILL license and that you accept its terms.           #
############################################################################
 */

package generator.coalescencemodel;

import java.util.ArrayList;
import java.util.HashMap;

import cycle.FullCyclicityTest;
import generator.IParasiteGenerator;
import generator.IScenario;
import graph.Edge;
import trees.Tree;
import trees.TreeNode;

public class CoalescenceModel implements IParasiteGenerator {

    /** Enumeration which defines the three possible events. */
    private static enum Events {
        COSPECIATION, DUPLICATION, HOST_SWITCH
    };


    private static double probabilityToChooseFirstInThePath = 0.5;

    /** Host tree that will be used in the parasite generation procedure. */
    private Tree hostTree;

    /** Node rho (fake root tree) for the host tree. */
    private TreeNode rhoHostTree;

    /** Real parasite tree. */
    private Tree realParasiteTree;

    /** Co-speciation probability (received by parameter during construction of the object) */
    private double initialCospeciationProbability;

    /** Duplication probability (received by parameter during construction of the object) */
    private double initialDuplicationProbability;

    /** Duplication probability (received by parameter during construction of the object) */
    private double initialHostSwitchProbability;

    /** Co-speciation probability */
    private double cospeciationProbability;

    /** Duplication probability */
    private double duplicationProbability;

    /** Probability of Co-speciation + duplication */
    private double cospeciationPlusDuplication;

    /** Host switch probability. */
    private double hostSwitchProbability;

    /** Expected number of cospeciations. */
    private double expectedNumberOfCospeciations;

    /** Expected number of duplications. */
    private double expectedNumberOfDuplications;

    /** Expected number of host-switches. */
    private double expectedNumberOfHostSwitches;

    /**
     * Flag which determines if we are going to use the known associations (true) or generate random
     * associations (false).
     */
    private boolean useKnownAssociations;

    /**
     * Flag which determines if we are going to use the information about host node distance (true)
     * or not (false).
     */
    private boolean useDistances;

    /**
     * HashMap which keeps the function that maps parasite leaf nodes into host leaf nodes.
     */
    private HashMap<String, String> leafMapping;

    /**
     * Cyclicity test identifier.
     */
    private int cyclicityTestModel;


    /**
     * Constructor.
     * 
     * @param hostTree
     *        Host tree.
     * @param realParasiteTree
     *        Real parasite tree.
     * @param leafMapping
     *        HashMap which keeps the function that maps parasite leaf nodes into host leaf nodes.
     * @param cospeciationProbability
     *        Probability of cospeciation.
     * @param duplicationProbability
     *        Probability of duplication.
     * @param useKnownAssociations
     *        If true, the simulation model will consider the known associations among parasite and
     *        host leaves. Otherwise, random associations are going to be created.
     * @param useDistances
     *        If true, the simulation model will consider the distance between host nodes to give
     *        probabilities to the nodes during the randm selection process. Otherwise, the nodes
     *        are choosen with equal probabilities.
     * @param cyclicityTestModel
     *        Identifier of the model which is going to be used to test for cycles in the generated
     *        scenarios.
     */
    public CoalescenceModel(Tree hostTree,
                            Tree realParasiteTree,
                            HashMap<String, String> leafMapping,
                            double cospeciationProbability,
                            double duplicationProbability,
                            boolean useKnownAssociations,
                            boolean useDistances,
                            int cyclicityTestModel) {

        /* Set the host tree. */
        this.hostTree = hostTree;
        this.rhoHostTree = hostTree.getRoot();

        /* Set the real parasite tree. */
        this.realParasiteTree = realParasiteTree;

        /* Set the initial probability values. */
        initialCospeciationProbability = cospeciationProbability;
        initialDuplicationProbability = duplicationProbability;
        initialHostSwitchProbability = 1.0 - (initialCospeciationProbability + initialDuplicationProbability);

        /* Set the flag useKnownAssociations. */
        this.useKnownAssociations = useKnownAssociations;

        /* Set the flag useDistances. */
        this.useDistances = useDistances;

        /* Set the desired cyclicity test */
        this.cyclicityTestModel = cyclicityTestModel;

        /*
         * Set the HashMap which keeps the function that maps parasite leaf nodes into host leaf
         * nodes.
         */
        this.leafMapping = leafMapping;

    }


    /**
     * Generates a parasite tree.
     * 
     * @return An object that contains the generated parasite tree.
     */
    public IScenario generateParasiteTree() {

        /* Update the working values */
        cospeciationProbability = initialCospeciationProbability;
        duplicationProbability = initialDuplicationProbability;
        cospeciationPlusDuplication = cospeciationProbability + duplicationProbability;
        hostSwitchProbability = initialHostSwitchProbability;

        /* Compute the expected number of events. */
        int internalNodes = realParasiteTree.getNumberOfLeafNodes() - 1;
        expectedNumberOfCospeciations = cospeciationProbability * internalNodes;
        expectedNumberOfDuplications = duplicationProbability * internalNodes;
        expectedNumberOfHostSwitches = hostSwitchProbability * internalNodes;

        /* Create an empty scenario. */
        CoalescenceModelScenario scenario = new CoalescenceModelScenario();

        /* Init the leaf set. */
        scenario.initLeafSet(hostTree, realParasiteTree, leafMapping, useKnownAssociations);

        /* Repeat the process while we have more than one tree into the forest. */
        while (scenario.getForestSize() > 1) {
            event(scenario);
        }

        /* Set the root node of the generated parasite tree. */
        scenario.setRootNode();

        Tree parasiteTree = scenario.getParasiteTree();
        parasiteTree.updateDescendantList();
        HashMap<Integer, Integer> mapping = scenario.getParasiteToHostMapping();
        FullCyclicityTest cyclicityTest = new FullCyclicityTest(cyclicityTestModel,
                                                                hostTree,
                                                                parasiteTree,
                                                                mapping);

        for (Edge transferEdge: scenario.getTransferEdges()) {
            TreeNode u = parasiteTree.getNode(transferEdge.getSource());
            TreeNode v = parasiteTree.getNode(transferEdge.getTarget());
            TreeNode gammaU = hostTree.getNode(mapping.get(u.getKey()));
            TreeNode gammaV = hostTree.getNode(mapping.get(v.getKey()));
            cyclicityTest.addTransferEdge(u, v, gammaU, gammaV);
        }

        if (cyclicityTest.getGraph().isCyclic()) {
            return null;
        }

        return scenario;

    }


    /**
     * Update the event probability values (must be called after updating the expected number of
     * events).
     */
    private void updateProbabilities() {
        double total = expectedNumberOfCospeciations + expectedNumberOfDuplications
                       + expectedNumberOfHostSwitches;
        cospeciationProbability = expectedNumberOfCospeciations / total;
        duplicationProbability = expectedNumberOfDuplications / total;
        hostSwitchProbability = expectedNumberOfHostSwitches / total;
        cospeciationPlusDuplication = cospeciationProbability + duplicationProbability;
    }


    /**
     * Throws a coin and returns one of the three possible events according with their
     * probabilities.
     * 
     * @return One of the three possible events: Co-speciation, Duplication and Host-Switch.
     */
    private Events randomEvent() {

        double coin = Math.random();

        if (coin < cospeciationProbability)
            return Events.COSPECIATION;

        if (coin < cospeciationPlusDuplication)
            return Events.DUPLICATION;

        return Events.HOST_SWITCH;

    }


    /**
     * Performs a random choice and register (if possible) the event into the scenario.
     * 
     * @param scenario
     *        Object which keeps control of the simulated scenario.
     */
    private void event(CoalescenceModelScenario scenario) {

        switch (randomEvent()) {
            case COSPECIATION:
                cospeciation(scenario);
                break;
            case DUPLICATION:
                duplication(scenario, true);
                break;
            case HOST_SWITCH:
                hostSwitch(scenario);
                break;
        }
    }


    /**
     * Register a co-speciation event into the scenario (if possible - otherwise, performs a
     * duplication).
     * 
     * @param scenario
     *        Object which keeps control of the simulated scenario.
     */
    private void cospeciation(CoalescenceModelScenario scenario) {

        Candidate selected = selectAnIncomparableCandidate(scenario);

        if (selected != null) {

            /* We can perform a co-speciation. */

            /* Register the associations. */
            TreeNode father = scenario.createFather(selected.p1, selected.p2);
            scenario.cospeciation(father, selected.lca);

            /* Update the number of losses. */
            int lossesP1 = hostTree.getPathFromDescendantToAncestor(selected.h1.getKey(),
                                                                    selected.lca.getKey()).size() - 2;
            int lossesP2 = hostTree.getPathFromDescendantToAncestor(selected.h2.getKey(),
                                                                    selected.lca.getKey()).size() - 2;
            scenario.loss(lossesP1 + lossesP2);

            /* Remove a co-speciation from the expected number of co-speciations. */
            expectedNumberOfCospeciations--;
            if (expectedNumberOfCospeciations < 0) {
                expectedNumberOfCospeciations = 0;
            }

        } else {

            /* We cannot perform a co-speciation, we are obliged to make a duplication. */

            duplication(scenario, false);
            if (expectedNumberOfDuplications > 0) {
                /* Remove a duplication from the expected number of duplications. */
                expectedNumberOfDuplications--;
                if (expectedNumberOfDuplications < 0) {
                    expectedNumberOfDuplications = 0;
                }
            } else {
                /* The duplications were all used, so remove from the co-speciation count. */
                expectedNumberOfCospeciations--;
                if (expectedNumberOfCospeciations < 0) {
                    expectedNumberOfCospeciations = 0;
                }
            }

        }

        updateProbabilities();

    }


    /**
     * Registers a duplication event into the scenario.
     * 
     * @param scenario
     *        Object which keeps control of the simulated scenario.
     * @param update
     *        If true, the method updates the event probabilities. Otherwise, the method let the
     *        update process for the one that called it.
     */
    private void duplication(CoalescenceModelScenario scenario, boolean update) {

        /* Copy of the current forest. */
        ArrayList<TreeNode> forest = new ArrayList<TreeNode>(scenario.getForest());

        /* Variables to keep the selected mapping. */
        TreeNode p1 = forest.remove((int)(Math.random() * forest.size()));
        TreeNode h1 = hostTree.getNode(scenario.getParasiteToHostMapping().get(p1.getKey()));
        Candidate selected = null;

        if (!useDistances) {

            TreeNode p2 = forest.remove((int)(Math.random() * forest.size()));
            TreeNode h2 = hostTree.getNode(scenario.getParasiteToHostMapping().get(p2.getKey()));
            TreeNode lca = hostTree.getLCA(h1.getKey(), h2.getKey());
            selected = new Candidate(0.0, p1, p2, h1, h2, lca);

        } else {

            /* Generate the list of candidates for a duplication. */
            ArrayList<Candidate> candidates = new ArrayList<Candidate>();
            for (TreeNode p2: forest) {
                TreeNode h2 = hostTree.getNode(scenario.getParasiteToHostMapping().get(p2.getKey()));
                TreeNode lca = hostTree.getLCA(h1.getKey(), h2.getKey());
                int distance = 1 + h2.getHeight() + h1.getHeight() - (2 * lca.getHeight());
                Candidate candidate = new Candidate(distance, p1, p2, h1, h2, lca);
                candidates.add(candidate);
            }

            /* Select one element from the list. */
            selected = selectCandidate(candidates);

        }

        /* Get all nodes from the path between the lca and the root of the host tree. */
        ArrayList<TreeNode> parents = hostTree.getPathFromDescendantToAncestor(selected.lca.getKey(),
                                                                               hostTree.getRoot().getKey());

        if (selected.lca != rhoHostTree && selected.lca != selected.h1
            && selected.lca != selected.h2) {
            /*
             * In this case, the lca must be removed from the possibilities (mapping into the lca
             * would be a co-speciation).
             */
            parents.remove(selected.lca);
        }

        /*
         * The choice is made based on the distance (50% for the closest, 50% for the rest,
         * recursively).
         */
        int index = 0;
        TreeNode positionToMap = null;
        while (index < parents.size() - 1) {
            double coin = Math.random();
            if (coin < probabilityToChooseFirstInThePath) {
                positionToMap = parents.get(index);
                break;
            }
            index++;
        }

        if (positionToMap == null) {
            /*
             * If selected is null, we passed through all other options and only remains the root of
             * the tree.
             */
            positionToMap = hostTree.getRoot();
        }

        /* Register the associations. */
        TreeNode father = scenario.createFather(selected.p1, selected.p2);
        scenario.duplication(father, positionToMap);

        /* Update the number of losses. */
        int lossesP1 = 0;
        if (selected.h1 != positionToMap)
            lossesP1 = hostTree.getPathFromDescendantToAncestor(selected.h1.getKey(),
                                                                positionToMap.getKey()).size() - 1;

        int lossesP2 = 0;
        if (selected.h2 != positionToMap)
            lossesP2 = hostTree.getPathFromDescendantToAncestor(selected.h2.getKey(),
                                                                positionToMap.getKey()).size() - 1;

        scenario.loss(lossesP1 + lossesP2);

        if (update) {
            /* Remove a duplication from the expected number of duplications. */
            expectedNumberOfDuplications--;
            if (expectedNumberOfDuplications < 0) {
                expectedNumberOfDuplications = 0;
            }
            updateProbabilities();
        }

    }


    /**
     * Registers a host-switch event into the scenario.
     * 
     * @param scenario
     *        Object which keeps control of the simulated scenario.
     */
    private void hostSwitch(CoalescenceModelScenario scenario) {

        Candidate selected = selectAnIncomparableCandidate(scenario);

        if (selected != null) {

            /* We can perform a host-switch */

            /*
             * Assumes that p1 stays and p2 jumps (no problem because it is always a random choice).
             */

            /* Get the list of candidates for mapping the new node. */
            ArrayList<TreeNode> candidates = hostTree.getPathFromDescendantToAncestor(selected.h1.getKey(),
                                                                                      selected.lca.getKey());
            /* Removes the lca from this list */
            candidates.remove(candidates.size() - 1);

            /*
             * The choice is made based on the distance (50% for the closest, 50% for the rest,
             * recursively).
             */
            int index = 0;
            TreeNode positionToMap = null;
            while (index < candidates.size() - 1) {
                double coin = Math.random();
                if (coin < probabilityToChooseFirstInThePath) {
                    positionToMap = candidates.get(index);
                    break;
                }
                index++;
            }

            if (positionToMap == null) {
                /*
                 * If selected is null, we passed through all other options and only remains the
                 * last node into the list.
                 */
                positionToMap = candidates.get(candidates.size() - 1);
            }

            /* Register the associations. */
            TreeNode father = scenario.createFather(selected.p1, selected.p2);
            scenario.hostSwitch(father, selected.p2, positionToMap);

            /* Update the number of losses. */
            int lossesP1 = 0;
            if (selected.h1 != positionToMap)
                lossesP1 = hostTree.getPathFromDescendantToAncestor(selected.h1.getKey(),
                                                                    positionToMap.getKey()).size() - 1;

            scenario.loss(lossesP1);

            /* Remove a host-switch from the expected number of host-switch. */
            expectedNumberOfHostSwitches--;
            if (expectedNumberOfHostSwitches < 0) {
                expectedNumberOfHostSwitches = 0;
            }

        } else {

            /* We cannot perform a host-switch, we are obliged to make a duplication. */
            duplication(scenario, false);
            if (expectedNumberOfDuplications > 0) {
                /* Remove a duplication from the expected number of duplications. */
                expectedNumberOfDuplications--;
                if (expectedNumberOfDuplications < 0) {
                    expectedNumberOfDuplications = 0;
                }
            } else {
                /* The duplications were all used, so remove from the co-speciation count. */
                expectedNumberOfHostSwitches--;
                if (expectedNumberOfHostSwitches < 0) {
                    expectedNumberOfHostSwitches = 0;
                }
            }

        }

        updateProbabilities();

    }


    /**
     * Select a candidate based on the distances.
     * 
     * @param candidates
     *        List of candidates
     * @param totalDistance
     *        Sum of all distances.
     * @return The selected candidate or null, if the list is empty.
     */
    private Candidate selectCandidate(ArrayList<Candidate> candidates) {

        if (candidates.size() > 0) {

            double totalDistance = 0.0;
            int power = candidates.size() * 2;
            for (Candidate candidate: candidates) {
                candidate.distance = Math.pow(candidate.distance, power);
                totalDistance += candidate.distance;
            }

            double sumWeight = 0.0;
            for (Candidate candidate: candidates) {
                candidate.weight = -Math.log(candidate.distance / totalDistance);
                sumWeight += candidate.weight;
            }

            double coin = Math.random();
            for (Candidate candidate: candidates) {
                double probability = candidate.weight / sumWeight;
                if (coin < probability) {
                    return candidate;
                } else {
                    coin -= probability;
                }
            }

            return candidates.get(candidates.size() - 1);

        }

        return null;

    }


    /**
     * Explore the current forest and select a pair of parasite nodes which are mapped into host
     * incomparable nodes.
     * 
     * @param scenario
     *        The current scenario.
     * @return A Candidate object with the necessary information about the nodes which were found or
     *         null, if there is no possibility.
     */
    private Candidate selectAnIncomparableCandidate(CoalescenceModelScenario scenario) {

        /* Copy of the current forest. */
        ArrayList<TreeNode> forest1 = new ArrayList<TreeNode>(scenario.getForest());

        if (!useDistances) {

            /*
             * Explore and return the candidate related to the first pair of incomparable host
             * nodes.
             */
            while (forest1.size() != 0) {

                /* Remove the node p1 from the forest. */
                TreeNode p1 = forest1.remove((int)(Math.random() * forest1.size()));
                TreeNode h1 = hostTree.getNode(scenario.getParasiteToHostMapping().get(p1.getKey()));

                /* Copy of forest1 after removing p1. */
                ArrayList<TreeNode> forest2 = new ArrayList<TreeNode>(forest1);

                while (forest2.size() != 0) {

                    /* Remove the node p2 from the forest. */
                    TreeNode p2 = forest2.remove((int)(Math.random() * forest2.size()));
                    TreeNode h2 = hostTree.getNode(scenario.getParasiteToHostMapping().get(p2.getKey()));

                    TreeNode lca = hostTree.getLCA(h1.getKey(), h2.getKey());

                    if (lca != h1 && lca != h2) {
                        /* The nodes are incomparable. */
                        return new Candidate(0.0, p1, p2, h1, h2, lca);
                    }

                }

            }

        } else {

            Candidate selected = null;
            boolean keepSearching = true;

            /* Explore all combinations, looking for a possible co-speciation. */
            while (forest1.size() != 0 && keepSearching) {

                /* Remove the node p1 from the forest. */
                TreeNode p1 = forest1.remove((int)(Math.random() * forest1.size()));
                TreeNode h1 = hostTree.getNode(scenario.getParasiteToHostMapping().get(p1.getKey()));

                /*
                 * Here we do not need to make a copy of forest 1, we are going to pass through all
                 * remaining elements.
                 */

                /* Get the list of candidates for a possible co-speciation. */
                ArrayList<Candidate> candidates = new ArrayList<Candidate>();
                for (TreeNode p2: forest1) {
                    TreeNode h2 = hostTree.getNode(scenario.getParasiteToHostMapping().get(p2.getKey()));
                    TreeNode lca = hostTree.getLCA(h1.getKey(), h2.getKey());
                    if (lca != h1 && lca != h2) {
                        /* The nodes are incomparable, we can apply a co-speciation */
                        int distance = 1 + h2.getHeight() + h1.getHeight() - (2 * lca.getHeight());
                        Candidate candidate = new Candidate(distance, p1, p2, h1, h2, lca);
                        candidates.add(candidate);
                    }
                }

                /* Select one element from the list. */
                selected = selectCandidate(candidates);
                keepSearching = selected == null;

            }

            return selected;
        }

        return null;
    }


    protected class Candidate {

        protected double distance;

        protected TreeNode p1;

        protected TreeNode p2;

        protected TreeNode h1;

        protected TreeNode h2;

        protected TreeNode lca;

        protected double weight;


        protected Candidate(double distance,
                            TreeNode p1,
                            TreeNode p2,
                            TreeNode h1,
                            TreeNode h2,
                            TreeNode lca) {
            this.distance = distance;
            this.p1 = p1;
            this.p2 = p2;
            this.h1 = h1;
            this.h2 = h2;
            this.lca = lca;
        }


        public String toString() {
            return weight + "\t" + distance + "\t" + p1.getLabel() + "\t" + p2.getLabel() + "\t"
                   + h1.getLabel() + "\t" + h2.getLabel() + "\t" + lca.getLabel();
        }

    }

}
