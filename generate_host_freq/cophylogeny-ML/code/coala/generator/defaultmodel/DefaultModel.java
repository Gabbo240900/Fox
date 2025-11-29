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

package generator.defaultmodel;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Random;

import cycle.IncrementalCyclicityTest;
import generator.IParasiteGenerator;
import generator.IScenario;
import graph.Edge;
import trees.Tree;
import trees.TreeNode;

public class DefaultModel implements IParasiteGenerator {

    /**
     * Random generator.
     */
    private Random random;

    /**
     * Host tree that will be used in the parasite generation procedure.
     */
    private Tree hostTree;

    /**
     * Co-speciation probability
     */
    private double cospeciationProbability = 0.0;

    /**
     * Duplication probability
     */
    private double duplicationProbability = 0.0;

    /**
     * Host switch probability.
     */
    private double hostSwitchProbability = 0.0;

    /**
     * Loss probability.
     */
    private double lossProbability = 0.0;

    /**
     * Maximum number of leaves for the generated tree.
     */
    private int maximumNumberOfLeaves;

    /**
     * Cyclicity test identifier.
     */
    private int cyclicityTestModel;

    /**
     * Minimum number of co-speciation events.
     */
    private int minCospeciation;

    /**
     * Minimum number of duplication events.
     */
    private int minDuplication;

    /**
     * Minimum number of host-switch events.
     */
    private int minHostSwitch;

    /**
     * Minimum number of loss events.
     */
    private int minLoss;

    /**
     * Maximum number of co-speciation events.
     */
    private int maxCospeciation;

    /**
     * Maximum number of duplication events.
     */
    private int maxDuplication;

    /**
     * Maximum number of host-switch events.
     */
    private int maxHostSwitch;

    /**
     * Maximum number of loss events.
     */
    private int maxLoss;

    /**
     * Factor used to compute mimimum and maximum number of expected evetns.
     */
    private double expectedFactor = 0.05;

    /**
     * Flag that indicates if we are going to use the maximum size stop criteria (true) or no.
     */
    private boolean maximumSizeStopCriterium;

    /**
     * Probability of mapping into the root node.
     */
    private double rootMappingProbability;


    /**
     * Constructor.
     * 
     * @param hostTree
     *        Host tree
     * @param cospeciationProbability
     *        Probability of cospeciation.
     * @param duplicationProbability
     *        Probability of duplication.
     * @param hostSwitchProbability
     *        Probability of host switch.
     * @param maximumNumberOfLeaves
     *        Maximum number of leaves for the generated tree.
     * @param cyclicityTestModel
     *        Identifier of the model which is going to be used to test for cycles in the generated
     *        scenarios.
     * @param rootMappingProbability
     *        Probability of mapping into the root node.
     */
    public DefaultModel(Tree hostTree,
                        double cospeciationProbability,
                        double duplicationProbability,
                        double hostSwitchProbability,
                        int numberOfLeavesRealParasite,
                        int maximumNumberOfLeaves,
                        int cyclicityTestModel,
                        boolean maximumSizeStopCriterium,
                        double rootMappingProbability) {

        /* Set the host tree. */
        this.hostTree = hostTree;

        /* Fill the main parameter set. */
        this.cospeciationProbability = cospeciationProbability;
        this.duplicationProbability = duplicationProbability;
        this.hostSwitchProbability = hostSwitchProbability;
        this.lossProbability = 1.0 - (cospeciationProbability + duplicationProbability + hostSwitchProbability);
        this.maximumNumberOfLeaves = maximumNumberOfLeaves;

        /* Cyclicity Test Model Identifier. */
        this.cyclicityTestModel = cyclicityTestModel;

        /* Maximum size stop criterium */
        this.maximumSizeStopCriterium = maximumSizeStopCriterium;

        /* Probability of mapping into the root node. */
        this.rootMappingProbability = rootMappingProbability;

        /* Create the random generator */
        this.random = new Random();

        int nInternal = numberOfLeavesRealParasite - 1;
        double cds = cospeciationProbability + duplicationProbability + hostSwitchProbability;
        double pc = cospeciationProbability / cds;
        double pd = duplicationProbability / cds;
        double ps = hostSwitchProbability / cds;

        double nTotal = nInternal / cds;
        double expectedNumberOfCospeciations = nInternal * pc;
        double expectedNumberOfDuplications = nInternal * pd;
        double expectedNumberOfHostSwitches = nInternal * ps;
        double expectedNumberOfLosses = nTotal * lossProbability;

        minCospeciation = (int)Math.floor((1.0 - expectedFactor) * expectedNumberOfCospeciations);
        minDuplication = (int)Math.floor((1.0 - expectedFactor) * expectedNumberOfDuplications);
        minHostSwitch = (int)Math.floor((1.0 - expectedFactor) * expectedNumberOfHostSwitches);
        minLoss = (int)Math.floor((1.0 - expectedFactor) * expectedNumberOfLosses);

        maxCospeciation = (int)Math.ceil((1.0 + expectedFactor) * expectedNumberOfCospeciations);
        maxDuplication = (int)Math.ceil((1.0 + expectedFactor) * expectedNumberOfDuplications);
        maxHostSwitch = (int)Math.ceil((1.0 + expectedFactor) * expectedNumberOfHostSwitches);
        maxLoss = (int)Math.ceil((1.0 + expectedFactor) * expectedNumberOfLosses);

    }


    /**
     * Chooses a start position for the simulation.
     * 
     * @param parasiteRootNode
     *        The parasite tree root node.
     * @return The choosen position.
     */
    private DefaultModelPosition chooseStartPosition(TreeNode parasiteRootNode) {

        TreeNode root = hostTree.getRoot();

        /* If forceRootToRoot is true, simply put the parasite node on the fake edge. */
        if (rootMappingProbability == 1.0)
            return new DefaultModelPosition(parasiteRootNode,
                                            new Edge(root.getKey(), root.getChild(0).getKey()),
                                            true,
                                            true,
                                            false);

        /* If not, we use the recursive procedure. */
        return chooseStartPosition(root.getChild(0), parasiteRootNode);
    }


    /**
     * Chooses a start position for the simulation.
     * 
     * @param subTreeRootNode
     *        Root of the candidate host subtree.
     * @param parasiteRootNode
     *        The parasite tree root node.
     * @return The choosen position.
     */
    private DefaultModelPosition chooseStartPosition(TreeNode subTreeRootNode,
                                                     TreeNode parasiteRootNode) {

        double coin = random.nextDouble();
        if (coin < rootMappingProbability || subTreeRootNode.isLeaf()) {
            return new DefaultModelPosition(parasiteRootNode,
                                            new Edge(subTreeRootNode.getParent().getKey(),
                                                     subTreeRootNode.getKey()),
                                            true,
                                            subTreeRootNode.getParent().getParent() != null,
                                            false);
        }

        TreeNode[] children = subTreeRootNode.getChildren();
        double numberOfLeavesLeft = children[0].getLeafSet().size();
        double numberOfLeavesRight = children[1].getLeafSet().size();
        double probabilityLeft = numberOfLeavesLeft / (numberOfLeavesLeft + numberOfLeavesRight);
        coin = random.nextDouble();
        if (coin < probabilityLeft) {
            return chooseStartPosition(children[0], parasiteRootNode);
        }
        return chooseStartPosition(children[1], parasiteRootNode);

    }


    /**
     * Generates a parasite tree.
     * 
     * @return An object that contains the generated parasite tree.
     */
    public IScenario generateParasiteTree() {

        /* Create an empty scenario. */
        DefaultModelScenario scenario = new DefaultModelScenario();

        /* Create the cyclicity test object. */
        IncrementalCyclicityTest cyclicityTest = new IncrementalCyclicityTest(cyclicityTestModel,
                                                                              hostTree,
                                                                              scenario.getParasiteTree(),
                                                                              scenario.getParasiteToHostMapping());

        /* Queue of positions (parasite node -> host edge). */
        LinkedList<DefaultModelPosition> queue = new LinkedList<DefaultModelPosition>();

        /* Choose a start position. */
        queue.add(chooseStartPosition(scenario.getRootNode()));

        /*
         * Associate the root of the parasite tree with the fake edge (start position) and add this
         * association (position) into the queue. Mark it as loss forbidden to force the mapping to
         * the root node.
         */

        HashMap<DefaultModelPosition, Edge> mandatoryEdges = new HashMap<DefaultModelPosition, Edge>();
        HashMap<DefaultModelPosition, DefaultModelPosition> duplicationPositions = new HashMap<DefaultModelPosition, DefaultModelPosition>();

        /* While the queue is not empty, evolve the parasite tree. */
        while (!queue.isEmpty()) {

            /* Remove the first position of the queue. */
            DefaultModelPosition currentPosition = queue.removeFirst();

            /* Get its sister position or mandatory edge (if they exist) */
            Edge mandatoryEdge = null;
            DefaultModelPosition sisterPosition = null;
            if (duplicationPositions.containsKey(currentPosition)) {
                sisterPosition = duplicationPositions.remove(currentPosition);
            } else if (mandatoryEdges.containsKey(currentPosition)) {
                mandatoryEdge = mandatoryEdges.remove(currentPosition);
            }

            /* Get parasite node and host edge */
            Edge e = currentPosition.getHostEdge();
            TreeNode edgeTarget = hostTree.getNode(e.getTarget());
            TreeNode n = currentPosition.getParasiteNode();

            if (edgeTarget.isLeaf()) {

                /*
                 * We have to verify if the frequencies of duplication and host-switch had not
                 * passed the probability vector.
                 */
                boolean allowDuplication = scenario.getFrequencyOfDuplications() < duplicationProbability;
                boolean allowHostSwitch = scenario.getFrequencyOfHostSwitches() < hostSwitchProbability;
                if (allowDuplication || allowHostSwitch) {

                    double stopProbability = cospeciationProbability + lossProbability;
                    if (!allowDuplication) {
                        stopProbability += duplicationProbability;
                    }
                    if (!allowHostSwitch) {
                        stopProbability += hostSwitchProbability;
                    }

                    double coin = random.nextDouble();
                    if (coin < stopProbability) {
                        stop(n, e, scenario);
                    } else {

                        if (allowDuplication && coin < stopProbability + duplicationProbability) {
                            /* In this case, we perform a duplication. */
                            duplicationInLeafEdge(n, e, queue, scenario);
                        } else if (allowHostSwitch) {
                            /* In this case, we perform a host switch. */
                            if (!hostSwitch(n, e, cyclicityTest, queue, scenario)) {
                                /* In this case we perform a STOP operation. */
                                stop(n, e, scenario);
                            }
                        } else {
                            stop(n, e, scenario);
                        }

                    }

                } else {

                    /* We cannot do anything but a stop. */
                    stop(n, e, scenario);

                }

            } else {

                /*
                 * We are in an inner edge, any of the four operations are allowed.
                 */

                double cospeciationRangeLimit = cospeciationProbability;
                double duplicationRangeLimit = cospeciationProbability + duplicationProbability;
                double transferRangeLimit = (cospeciationProbability + duplicationProbability + hostSwitchProbability);

                boolean operationPerformed = false;
                while (!operationPerformed) {

                    /* Adjust the probabilities if necessary. */
                    if (currentPosition.isLossForbidden()) {
                        if (currentPosition.isHostSwitchForbidden()) {
                            double withoutLossAndHostSwitch = (cospeciationProbability + duplicationProbability);
                            cospeciationRangeLimit = (cospeciationProbability / withoutLossAndHostSwitch);
                            duplicationRangeLimit = 1.0;
                            transferRangeLimit = 1.0;
                        } else {
                            double withoutLoss = (cospeciationProbability + duplicationProbability + hostSwitchProbability);
                            cospeciationRangeLimit = cospeciationProbability / withoutLoss;
                            duplicationRangeLimit = (cospeciationRangeLimit + (duplicationProbability / withoutLoss));
                            transferRangeLimit = 1.0;
                        }
                    } else if (currentPosition.isHostSwitchForbidden()) {
                        double withoutHostSwitch = (cospeciationProbability
                                                    + duplicationProbability + lossProbability);
                        cospeciationRangeLimit = (cospeciationProbability / withoutHostSwitch);
                        duplicationRangeLimit = (cospeciationRangeLimit + (duplicationProbability / withoutHostSwitch));
                        transferRangeLimit = duplicationRangeLimit;
                    }

                    /* Perform the operation. */
                    double coin = random.nextDouble();
                    if (coin < cospeciationRangeLimit) {
                        /* COSPECIATION operation. */
                        cospeciation(n, e, queue, scenario);
                        operationPerformed = true;
                    } else if (coin < duplicationRangeLimit) {
                        /* DUPLICATION operation. */
                        duplication(n, e, queue, duplicationPositions, scenario);
                        operationPerformed = true;
                    } else if (coin < transferRangeLimit) {
                        /* TRANSFER operation. */
                        if (hostSwitch(n, e, cyclicityTest, queue, scenario)) {
                            operationPerformed = true;
                        } else {
                            currentPosition.setHostSwitchForbiddenTrue();
                        }
                    } else {
                        /* LOSS operation. */
                        loss(n, e, queue, sisterPosition, mandatoryEdge, mandatoryEdges, scenario);
                        operationPerformed = true;
                    }

                }

            }

            if (maximumSizeStopCriterium) {
                if (scenario.getNumberOfLeaves() > maximumNumberOfLeaves) {
                    return null;
                }
            } else {
                if (scenario.getNumberOfCospeciations() > maxCospeciation
                    || scenario.getNumberOfDuplications() > maxDuplication
                    || scenario.getNumberOfHostSwitches() > maxHostSwitch
                    || scenario.getNumberOfLosses() > maxLoss) {
                    return null;
                }
            }

        }

        if (!maximumSizeStopCriterium) {
            if (scenario.getNumberOfCospeciations() < minCospeciation
                || scenario.getNumberOfDuplications() < minDuplication
                || scenario.getNumberOfHostSwitches() < minHostSwitch
                || scenario.getNumberOfLosses() < minLoss) {
                return null;
            }
        }

        return scenario;

    }


    /**
     * Cospeciation event. Associate the parasite node to the head of the host edge. Create two new
     * children for the parasite node. Position one child of the parasite node on the left outgoing
     * edge of the head of the host edge and position the other child on the right outgoing edge of
     * the head of the host edge.
     * 
     * @param parasiteNode
     *        Parasite node.
     * @param hostEdge
     *        Host edge where the parasite node is positioned.
     * @param queue
     *        Queue of positions.
     * @param scenario
     *        Object that keeps the scenario that is being built.
     */
    private void cospeciation(TreeNode parasiteNode,
                              Edge hostEdge,
                              LinkedList<DefaultModelPosition> queue,
                              DefaultModelScenario scenario) {
        TreeNode target = hostTree.getNode(hostEdge.getTarget());
        TreeNode[] children = scenario.createChildren(parasiteNode);
        scenario.cospeciation(parasiteNode, target);
        queue.add(new DefaultModelPosition(children[0],
                                           new Edge(target.getKey(), target.getChild(0).getKey()),
                                           false,
                                           false,
                                           false));
        queue.add(new DefaultModelPosition(children[1],
                                           new Edge(target.getKey(), target.getChild(1).getKey()),
                                           false,
                                           false,
                                           false));
    }


    /**
     * Duplication event. Associate the parasite node to the tail of the host edge. Create two new
     * children for the parasite node. Position the two children of the parasite node on the host
     * edge.
     * 
     * @param parasiteNode
     *        Parasite node.
     * @param hostEdge
     *        Host edge where the parasite node is positioned.
     * @param queue
     *        Queue of positions.
     * @param duplicationPositions
     *        HashMap that will keep the association between the two positions of the two children
     *        that were produced during the duplication.
     * @param scenario
     *        Object that keeps the scenario that is being built.
     */
    private void duplication(TreeNode parasiteNode,
                             Edge hostEdge,
                             LinkedList<DefaultModelPosition> queue,
                             HashMap<DefaultModelPosition, DefaultModelPosition> duplicationPositions,
                             DefaultModelScenario scenario) {

        TreeNode source = hostTree.getNode(hostEdge.getSource());
        TreeNode target = hostTree.getNode(hostEdge.getTarget());
        TreeNode[] children = scenario.createChildren(parasiteNode);
        scenario.duplication(parasiteNode, target);

        /*
         * Both children are mapped in the host edge.
         */
        DefaultModelPosition position1 = new DefaultModelPosition(children[0],
                                                                  hostEdge,
                                                                  false,
                                                                  source.isRoot(),
                                                                  false);

        DefaultModelPosition position2 = new DefaultModelPosition(children[1],
                                                                  hostEdge,
                                                                  false,
                                                                  source.isRoot(),
                                                                  false);
        duplicationPositions.put(position1, position2);
        queue.add(position1);
        queue.add(position2);

    }


    /**
     * Duplication event in a leaf edge. Associate the parasite node to the tail of the host edge.
     * Create two new children for the parasite node. Position the two children of the parasite node
     * on the host edge and mark them with the flag FORCE STOP.
     * 
     * @param parasiteNode
     *        Parasite node.
     * @param hostEdge
     *        Host edge where the parasite node is positioned.
     * @param queue
     *        Queue of positions.
     * @param scenario
     *        Object that keeps the scenario that is being built.
     */
    private void duplicationInLeafEdge(TreeNode parasiteNode,
                                       Edge hostEdge,
                                       LinkedList<DefaultModelPosition> queue,
                                       DefaultModelScenario scenario) {
        TreeNode source = hostTree.getNode(hostEdge.getSource());
        TreeNode target = hostTree.getNode(hostEdge.getTarget());
        TreeNode[] children = scenario.createChildren(parasiteNode);
        scenario.duplication(parasiteNode, target);
        queue.add(new DefaultModelPosition(children[0], hostEdge, false, source.isRoot(), false));
        queue.add(new DefaultModelPosition(children[1], hostEdge, false, source.isRoot(), false));
    }


    /**
     * Host Switch operation. Search for a valid incomparable node. Create two children for the
     * parasite node. Associate the parasite node to the tail of the host edge. Position one of the
     * child on the host edge and the other on the incoming edge of the incomparable node.
     * 
     * @param parasiteNode
     *        Parasite node.
     * @param hostEdge
     *        Host edge where the parasite node is positioned.
     * @param cyclicityTest
     *        Cyclicity test model.
     * @param queue
     *        Queue of positions.
     * @param scenario
     *        Object that keeps the scenario that is being built.
     * @return TRUE if the host switch operation could be performed, FALSE otherwise.
     */
    private boolean hostSwitch(TreeNode parasiteNode,
                               Edge hostEdge,
                               IncrementalCyclicityTest cyclicityTest,
                               LinkedList<DefaultModelPosition> queue,
                               DefaultModelScenario scenario) {

        TreeNode hostNode = hostTree.getNode(hostEdge.getTarget());

        TreeNode incomparableNode = cyclicityTest.randomIncomparable(parasiteNode, hostNode);
        if (incomparableNode == null) {
            /* It is not possible to perform a transfer. */
            return false;
        }

        TreeNode[] children = scenario.createChildren(parasiteNode);

        double coin = random.nextDouble();
        if (coin < 0.5) {

            /* Children 0 stays in the same position where the host node is located. */
            /* Loss is not forbidden */
            /* Host switch is not forbidden. */
            /* Force stop only if hostNode is a leaf. */
            queue.add(new DefaultModelPosition(children[0], hostEdge, false, false, false));

            /* Children 1 is the parasite who jumped to another subtree. */
            /* Loss is forbidden */
            /* Host switch is not forbidden. */
            /* Force stop only if hostNode and incomparableNode are leaves. */
            queue.add(new DefaultModelPosition(children[1],
                                               /* Position in a incomparable branch. */
                                               new Edge(incomparableNode.getParent().getKey(),
                                                        incomparableNode.getKey()),
                                               true,
                                               false,
                                               false));

            scenario.hostSwitch(parasiteNode, children[1], hostNode, incomparableNode);

            cyclicityTest.addTransferEdge(parasiteNode, children[1], hostNode, incomparableNode);

        } else {

            /* Children 0 is the parasite who jumped to another subtree. */
            /* Loss is forbidden */
            /* Host switch is not forbidden. */
            /* Force stop only if hostNode and incomparableNode are leaves. */
            queue.add(new DefaultModelPosition(children[0],
                                               new Edge(incomparableNode.getParent().getKey(),
                                                        incomparableNode.getKey()),
                                               true,
                                               false,
                                               false));

            /* Children 1 stays in the same position where the host node is located. */
            /* Loss is not forbidden */
            /* Host switch is not forbidden. */
            /* Force stop only if hostNode is a leaf. */
            queue.add(new DefaultModelPosition(children[1], hostEdge, false, false, false));

            scenario.hostSwitch(parasiteNode, children[0], hostNode, incomparableNode);
            cyclicityTest.addTransferEdge(parasiteNode, children[0], hostNode, incomparableNode);

        }

        return true;
    }


    /**
     * LOSS operation. Position the parasite node to one of the outgoing edges of the host edge
     * head.
     * 
     * @param parasiteNode
     *        Parasite node.
     * @param hostEdge
     *        Host edge where the parasite node is positioned.
     * @param queue
     *        Queue of positions.
     * @param sisterPosition
     *        If not null, it is the position of the slibbing node of the parasite node (which was
     *        produced during a duplication operation).
     * @param mandatoryEdge
     *        If not null, the node must follow the this edge.
     * @param mandatoryEdges
     *        If sisterPosition is not null, we have to register that the sisterPosition will be
     *        associated to a mandatory edge (in case of loss).
     * @param scenario
     *        Object that keeps the scenario that is being built.
     */
    private void loss(TreeNode parasiteNode,
                      Edge hostEdge,
                      LinkedList<DefaultModelPosition> queue,
                      DefaultModelPosition sisterPosition,
                      Edge mandatoryEdge,
                      HashMap<DefaultModelPosition, Edge> mandatoryEdges,
                      DefaultModelScenario scenario) {
        if (mandatoryEdge == null) {
            double coin = random.nextDouble();
            TreeNode target = hostTree.getNode(hostEdge.getTarget());
            Edge selectedEdge = null;
            if (coin < 0.5) {
                selectedEdge = new Edge(target.getKey(), target.getChild(0).getKey());
            } else {
                selectedEdge = new Edge(target.getKey(), target.getChild(1).getKey());
            }
            queue.add(new DefaultModelPosition(parasiteNode, selectedEdge, false, false, false));
            if (sisterPosition != null) {
                mandatoryEdges.put(sisterPosition, selectedEdge);
            }
        } else {
            queue.add(new DefaultModelPosition(parasiteNode, mandatoryEdge, false, false, false));
        }
        scenario.loss();
    }


    /**
     * STOP operation. Associate the parasite node with the host edge head (that is a leaf node).
     * 
     * @param parasiteNode
     *        Parasite node.
     * @param hostEdge
     *        Host edge where the parasite node is positioned.
     * @param scenario
     *        Object that keeps the scenario that is being built.
     */
    private void stop(TreeNode parasiteNode, Edge hostEdge, DefaultModelScenario scenario) {
        TreeNode target = hostTree.getNode(hostEdge.getTarget());
        scenario.stop(parasiteNode, target);

    }

}
