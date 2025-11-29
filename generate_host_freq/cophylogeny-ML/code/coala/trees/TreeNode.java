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

package trees;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class TreeNode {

    /** Children of this node. */
    private TreeNode[] children;

    /** Key of this node. */
    private int key;

    /** Label of this node. */
    private String label;

    /** Parent of this node. */
    private TreeNode parent;

    /** Node height. */
    private int height;

    /** List of ancestors of this node (including itself) */
    private HashSet<TreeNode> ancestors;

    /** List of descendants of this node (including itself) */
    private HashSet<TreeNode> descendants;


    /**
     * Creates a tree node with the given key.
     * 
     * @param key
     *        The node key.
     */
    protected TreeNode(int key) {
        this.key = key;
        this.children = new TreeNode[2];
        this.label = Integer.toString(key);
        this.height = 1;
        this.ancestors = new HashSet<TreeNode>();
        this.ancestors.add(this);
        this.descendants = new HashSet<TreeNode>();
        this.descendants.add(this);
    }


    /**
     * Creates a tree node with the given key and label.
     * 
     * @param key
     *        The node key.
     * @param label
     *        The node label.
     */
    protected TreeNode(int key, String label) {
        this.key = key;
        this.label = label;
        this.children = new TreeNode[2];
        this.height = 1;
    }


    /**
     * Adds a child for this tree node on the specified position.
     * 
     * @param child
     *        Child to be added.
     * @param position
     *        0 - left child, 1 - right child.
     */
    protected void addChild(TreeNode child, int position) {
        child.setParent(this);
        children[position] = child;
    }


    /**
     * Add an ancestor node into the list of ancestors.
     * 
     * @param node
     *        Node to be added.
     */
    public void addAncestor(TreeNode node) {
        ancestors.add(node);
    }


    /**
     * Add a descendant node into the list of descendants.
     * 
     * @param node
     *        Node to be added.
     */
    public void addDescendant(TreeNode node) {
        descendants.add(node);
    }


    /**
     * Empty the list of ancestors.
     */
    protected void emptyAncestors() {
        ancestors = new HashSet<TreeNode>();
    }


    /**
     * Empty the list of descendants.
     */
    protected void emptyDescendants() {
        descendants = new HashSet<TreeNode>();
    }


    /**
     * Add a list of ancestor nodes.
     * 
     * @param node
     *        Nodes to be added.
     */
    protected void addAncestors(HashSet<TreeNode> nodes) {
        ancestors.addAll(nodes);
    }


    /**
     * Add a list of descendant nodes.
     * 
     * @param node
     *        Nodes to be added.
     */
    protected void addDescendants(HashSet<TreeNode> nodes) {
        descendants.addAll(nodes);
    }


    /**
     * Returns the list of descendant nodes.
     * 
     * @return The list of descendant nodes.
     */
    protected HashSet<TreeNode> getDescendants() {
        return descendants;
    }


    /**
     * Returns the list of ancestor nodes.
     * 
     * @return The list of ancestor nodes.
     */
    protected HashSet<TreeNode> getAncestors() {
        return ancestors;
    }


    /**
     * Returns the leaf set of the subtree rooted in this tree node.
     * 
     * @return The leaf set of the subtree rooted in this tree node.
     */
    public ArrayList<TreeNode> getLeafSet() {
        ArrayList<TreeNode> descendantsLeaves = new ArrayList<TreeNode>();
        if (this.isLeaf()) {
            descendantsLeaves.add(this);
        } else {
            if (children[0] != null) {
                descendantsLeaves.addAll(children[0].getLeafSet());
            }
            if (children[1] != null) {
                descendantsLeaves.addAll(children[1].getLeafSet());
            }
        }
        return descendantsLeaves;
    }


    /**
     * Returns the child of this node that is in the given position.
     * 
     * @param position
     *        0 - left child, 1 - right child.
     * @return The child of this node that is in the given position.
     */
    public TreeNode getChild(int position) {
        return children[position];
    }


    /**
     * Returns the children of this node.
     * 
     * @return The children of this node.
     */
    public TreeNode[] getChildren() {
        return children;
    }


    /**
     * Returns the height of this tree node.
     * 
     * @return The height of this tree node.
     */
    public int getHeight() {
        return height;
    }


    /**
     * Returns the key of this tree node.
     * 
     * @return The key of this tree node.
     */
    public int getKey() {
        return key;
    }


    /**
     * Returns the label of this tree node.
     * 
     * @return The label of this tree node.
     */
    public String getLabel() {
        return label;
    }


    /**
     * Returns the number of children of this tree node object.
     * 
     * @return The number of children of this tree node object.
     */
    public int getNumberOfChildren() {
        int numberOfChildren = 0;
        if (children[0] != null)
            numberOfChildren++;
        if (children[1] != null)
            numberOfChildren++;
        return numberOfChildren;
    }


    /**
     * Returns the parent of this tree node.
     * 
     * @return The parent of this tree node.
     */
    public TreeNode getParent() {
        return parent;
    }


    /**
     * Returns a hash code for this object.
     * 
     * @return A hash code for this object.
     */
    public int hashCode() {
        return key;
    }


    /**
     * Returns true if this node is a leaf node and false, otherwise.
     * 
     * @return True if this node is a leaf node and false, otherwise.
     */
    public boolean isLeaf() {
        return (children[0] == null && children[1] == null);
    }


    /**
     * Returns true if this node is a root node and false, otherwise.
     * 
     * @return True if this node is a root node and false, otherwise.
     */
    public boolean isRoot() {
        return (null == parent);
    }


    /**
     * This method produce the ArrayList rapresentation of the partition of the node.
     */
    public ArrayList<Integer> partition(HashMap<String, Integer> leavesOrder) {
        int numberOfLeaves = leavesOrder.size();
        ArrayList<TreeNode> leafSet = getLeafSet();
        ArrayList<Integer> partition = new ArrayList<Integer>(numberOfLeaves);
        for (int i = 0; i < numberOfLeaves; i++) {
            partition.add(0);
        }
        for (TreeNode leaf: leafSet) {
            int index = leavesOrder.get(leaf.getLabel());
            int v = partition.get(index);
            partition.set(index, v + 1);
        }
        return partition;
    }


    /**
     * Sets the height of this node.
     * 
     * @param height
     *        A new height value for this node.
     */
    public void setHeight(int height) {
        this.height = height;
    }


    /**
     * Sets a new label for this tree node.
     * 
     * @param label
     *        A new label for this tree node.
     */
    public void setLabel(String label) {
        this.label = label;
    }


    /**
     * Sets the parent of this tree node.
     * 
     * @param parent
     *        Parent of this tree node object.
     */
    private void setParent(TreeNode parent) {
        this.parent = parent;
    }


    /**
     * Returns a string representation of this tree node.
     * 
     * @return A string representation of this tree node.
     */
    public String toString() {
        return key + ":" + label;
    }

}
