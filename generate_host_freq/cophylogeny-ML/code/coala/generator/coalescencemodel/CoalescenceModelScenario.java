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

//##############################################################################
//############Is the same thing of DefaultModelScenario but without loss########
//##############################################################################
package generator.coalescencemodel;

import java.util.ArrayList;
import java.util.HashMap;

import generator.Association;
import generator.ScenarioAbst;
import graph.Edge;
import trees.Tree;
import trees.TreeNode;

public class CoalescenceModelScenario extends ScenarioAbst {

    /**
     * This arraylist represent a forest of trees which are available for being joined together
     * during the coalescent process.
     */
    private ArrayList<TreeNode> forest;

    /** List of transfer edges. */
    private ArrayList<Edge> transferEdges;


    /**
     * Creates an empty scenario: no associations and an empty parasite tree (no nodes or edges).
     */
    public CoalescenceModelScenario() {
        super();
        initParasiteTree(false);
        this.forest = new ArrayList<TreeNode>();
        this.transferEdges = new ArrayList<Edge>();
    }


    /**
     * Adds an association that represents a co-speciation event.
     * 
     * @param parasiteNode
     *        The parasite node.
     * @param hostNode
     *        The host node.
     */
    public void cospeciation(TreeNode parasiteNode, TreeNode hostNode) {
        addAssociation(new Association(parasiteNode, hostNode, Association.COSPECIATION));
        parasiteToHost.put(parasiteNode.getKey(), hostNode.getKey());
        increaseCospeciationCounter();
    }


    /**
     * Add an association that represents a duplication event.
     * 
     * @param parasiteNode
     *        The parasite node.
     * @param hostNode
     *        The host node.
     */
    public void duplication(TreeNode parasiteNode, TreeNode hostNode) {
        addAssociation(new Association(parasiteNode, hostNode, Association.DUPLICATION));
        parasiteToHost.put(parasiteNode.getKey(), hostNode.getKey());
        increaseDuplicationCounter();
    }


    /**
     * Add an association that represents a host switch event.
     * 
     * @param parasiteNode
     *        The parasite node.
     * @param hostNode
     *        The host node.
     */
    public void hostSwitch(TreeNode parasiteNode, TreeNode transferedChild, TreeNode hostNode) {
        addAssociation(new Association(parasiteNode, hostNode, Association.HOSTSWITCH));
        parasiteToHost.put(parasiteNode.getKey(), hostNode.getKey());
        transferEdges.add(new Edge(parasiteNode.getKey(), transferedChild.getKey()));
        increaseHostSwitchCounter();
    }


    /**
     * Updates the number of losses.
     * 
     * @param numberOfLosses
     *        Total number of losses to be added to the scenario.
     */
    public void loss(int numberOfLosses) {
        increaseLossCounter(numberOfLosses);
    }


    /**
     * Returns the size of the forest of trees which is associated to this scenario.
     * 
     * @return The size of the forest of trees which is associated to this scenario.
     */
    public int getForestSize() {
        return forest.size();
    }


    /**
     * Returns the forest of trees which is associated to this scenario.
     * 
     * @return The forest of trees which is associated to this scenario.
     */
    public ArrayList<TreeNode> getForest() {
        return forest;
    }


    /**
     * Performs the initialisation of the leaf set of the parasite tree.
     * 
     * @param hostTree
     *        Host tree (base of the simulation).
     * @param realParasiteTree
     *        Real parasite tree.
     * @param leafMapping
     *        HashMap which keeps the function that maps parasite leaf nodes into host leaf nodes.
     * @param useKnownAssociations
     *        If true, the method uses the known associations among host and parasite leaves.
     *        Otherwise, the known associations are completelly ignored.
     */
    public void initLeafSet(Tree hostTree,
                            Tree realParasiteTree,
                            HashMap<String, String> leafMapping,
                            boolean useKnownAssociations) {

        Tree parasiteTree = getParasiteTree();
        TreeNode[] hostNodes = hostTree.getLeafNodes();
        TreeNode[] realParasiteNodes = realParasiteTree.getLeafNodes();

        int hostNumberOfLeaves = hostNodes.length;
        int realParasiteNumberOfLeaves = realParasiteNodes.length;

        if (useKnownAssociations) {

            HashMap<String, TreeNode> mapLabelToNode = hostTree.getMapLabelToNode();
            for (String parasiteLabel: leafMapping.keySet()) {
                String hostLabel = leafMapping.get(parasiteLabel);
                TreeNode pNode = parasiteTree.createNode(hostLabel);
                TreeNode hNode = mapLabelToNode.get(hostLabel);
                parasiteToHost.put(pNode.getKey(), hNode.getKey());
                forest.add(pNode);
            }

        } else {

            for (int i = 0; i < realParasiteNumberOfLeaves; i++) {
                TreeNode hNode = hostNodes[(int)(hostNumberOfLeaves * Math.random())];
                TreeNode pNode = parasiteTree.createNode(hNode.getLabel());
                forest.add(pNode);
                parasiteToHost.put(pNode.getKey(), hNode.getKey());
            }

        }

        increaseNumberOfLeaves(realParasiteNumberOfLeaves);

    }


    /**
     * Creates a new node and assign it as father of the two given nodes. The new node is added to
     * the forest and the two children are removed.
     * 
     * @param child1
     *        First child.
     * @param child2
     *        Second child.
     * @return Returns the created node.
     */
    public TreeNode createFather(TreeNode child1, TreeNode child2) {
        Tree parasiteTree = getParasiteTree();
        TreeNode father = parasiteTree.createNode();
        parasiteTree.addChild(father.getKey(), child1.getKey(), 0);
        parasiteTree.addChild(father.getKey(), child2.getKey(), 1);
        forest.remove(child1);
        forest.remove(child2);
        forest.add(father);
        return father;
    }


    /**
     * Set the root node of the parasite tree. This method should be called in the end of the
     * coalescent process.
     */
    public void setRootNode() {
        if (forest.size() != 1) {
            throw new RuntimeException("At this point, the forest should have only one tree.");
        }
        Tree parasiteTree = getParasiteTree();
        parasiteTree.setRoot(forest.get(0).getKey());
    }


    /**
     * Returns the list of transfer edges of the parasite tree.
     * 
     * @return The list of transfer edges of the parasite tree.
     */
    public ArrayList<Edge> getTransferEdges() {
        return transferEdges;
    }
}
