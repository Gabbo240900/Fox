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
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multisets;

public class Tree {

    private final char OPEN_BRACKET = '(';

    private final char CLOSE_BRACKET = ')';

    private final char CHILD_SEPARATOR = ',';

    /** Variable which keeps a counter for generating key identifier for new nodes. */
    private int key;

    /** Tree's height */
    private int height;

    /** Root of the tree. */
    private TreeNode root;

    /** List of nodes of the tree. */
    private ArrayList<TreeNode> nodes;

    /** List of leaf nodes of the tree. */
    private ArrayList<TreeNode> leaves;

    /** HashMap which keeps the mapping key -> TreeNode object for each created node. */
    private HashMap<Integer, TreeNode> mapKeyToNode;

    /** HashMap which keeps the mapping label -> TreeNode object for each created node. */
    private HashMap<String, TreeNode> mapLabelToNode;

    /** Table containing all LCA of every pair of leaf nodes of the tree. */
    private TreeNode[][] lcaTable;


    /**
     * Creates a tree object.
     * 
     * @param createRoot
     *        If true, creates a root node and sets it with height 1.
     */
    public Tree(boolean createRoot) {
        this.key = 0;
        this.mapKeyToNode = new HashMap<Integer, TreeNode>();
        if (createRoot)
            this.root = createNode();
        else
            this.root = null;
        this.height = -1;
        this.nodes = null;
        this.leaves = null;
        this.lcaTable = null;
        this.mapLabelToNode = null;

    }


    /**
     * Returns the node which has the given key.
     * 
     * @param key
     *        The node key.
     * @return The node which has the given key.
     */
    public TreeNode getNode(int key) {
        TreeNode node = mapKeyToNode.get(key);
        if (node == null)
            throw new RuntimeException("Invalid key identifier");
        return node;
    }


    /**
     * Returns the root of the tree.
     * 
     * @return The root of the tree.
     */
    public TreeNode getRoot() {
        return root;
    }


    /**
     * Set the root node of this tree.
     * 
     * @param key
     *        The key of the new root node.
     */
    public void setRoot(int key) {
        TreeNode node = mapKeyToNode.get(key);
        if (node == null)
            throw new RuntimeException("Invalid key identifier");
        if (node.getParent() != null)
            throw new RuntimeException("The root of the tree cannot have a parent node.");
        this.root = node;
        this.nodes = null;
        this.height = -1;
        this.leaves = null;
        this.lcaTable = null;
        this.mapLabelToNode = null;
    }


    /**
     * Creates a TreeNode object with unique identifier.
     * 
     * @return A TreeNode object with unique identifier.
     */
    public TreeNode createNode() {
        TreeNode node = new TreeNode(key++);
        mapKeyToNode.put(node.getKey(), node);
        return node;
    }


    /**
     * Creates a TreeNode object with unique identifier and the given label.
     * 
     * @return A TreeNode object with unique identifier and the given label.
     */
    public TreeNode createNode(String label) {
        TreeNode node = new TreeNode(key++, label);
        mapKeyToNode.put(node.getKey(), node);
        return node;
    }


    /**
     * Add a child into the given node at the give position.
     * 
     * @param node
     *        Node that will receive the child.
     * @param child
     *        The new child of the node.
     * @param position
     *        0 - left child; 1 - right child.
     */
    public void addChild(int node, int child, int position) {
        TreeNode parentNode = mapKeyToNode.get(node);
        TreeNode childNode = mapKeyToNode.get(child);
        if (parentNode != null && childNode != null)
            parentNode.addChild(childNode, position);
        this.nodes = null;
        this.height = -1;
        this.leaves = null;
        this.lcaTable = null;
        this.mapLabelToNode = null;
    }


    /**
     * Returns the height of the tree.
     * 
     * @return The height of the tree.
     */
    public int getHeight() {
        if (height == -1)
            updateAttributes();
        return height;
    }


    /**
     * Returns the total number of nodes of the tree (excluding eventual fake root).
     * 
     * @return The total number of nodes of the tree (excluding eventual fake root).
     */
    public int getNumberOfNodes() {
        if (nodes == null)
            updateAttributes();
        return nodes.size();
    }


    /**
     * Returns the total number of leaf nodes of the tree.
     * 
     * @return The total number of leaf nodes of the tree.
     */
    public int getNumberOfLeafNodes() {
        if (leaves == null)
            updateAttributes();
        return leaves.size();
    }


    /**
     * Traverses the tree in post-order and update the attributes of this tree (list of nodes, list
     * of leaves and tree height), excluding the eventual fake root.
     */
    private void updateAttributes() {
        this.nodes = new ArrayList<TreeNode>();
        this.leaves = new ArrayList<TreeNode>();
        this.height = -1;
        if (root == null)
            throw new RuntimeException("The tree does not have a root node.");
        if (root.getNumberOfChildren() == 1) {
            /* RHO. */
            updateAttributes(root, 0);
        } else {
            /* Real root. */
            updateAttributes(root, 1);
        }
    }


    /**
     * Recursive function that helps in the task of updating the attributes of this tree.
     * 
     * @param subtreeRoot
     *        Root of the subtree that is going to be traversed.
     * @param height
     *        The current height.
     */
    private void updateAttributes(TreeNode subtreeRoot, int height) {
        subtreeRoot.setHeight(height);
        if (subtreeRoot.isLeaf()) {
            leaves.add(subtreeRoot);
            if (this.height < height)
                this.height = height;
        } else {
            if (subtreeRoot.getChild(0) != null)
                updateAttributes(subtreeRoot.getChild(0), height + 1);
            if (subtreeRoot.getChild(1) != null)
                updateAttributes(subtreeRoot.getChild(1), height + 1);
        }
        nodes.add(subtreeRoot);
    }


    /**
     * Generate a string representing the traversing of the tree in post order. This method uses the
     * key values of the nodes.
     * 
     * @param node
     *        The root of the tree
     * @param addInternalNodes
     *        If <code>true</code>, the newick string includes the internal node labels. Otherwise,
     *        only leaf labels are present in the returned string.
     * @return A string representing the traversing of the tree in post order (using the key of the
     *         nodes).
     */
    private String postOrderStringKey(TreeNode node, boolean addInternalNodes) {
        if (node.isLeaf())
            return Integer.toString(node.getKey());
        if (node.getNumberOfChildren() == 2) {
            String ls = postOrderStringKey(node.getChild(0), addInternalNodes);
            String rs = postOrderStringKey(node.getChild(1), addInternalNodes);
            if (addInternalNodes)
                return OPEN_BRACKET + ls + CHILD_SEPARATOR + rs + CLOSE_BRACKET
                       + Integer.toString(node.getKey());
            else
                return OPEN_BRACKET + ls + CHILD_SEPARATOR + rs + CLOSE_BRACKET;
        } else if (node.getChild(0) != null) {
            String ls = postOrderStringKey(node.getChild(0), addInternalNodes);
            if (addInternalNodes)
                return OPEN_BRACKET + ls + CLOSE_BRACKET + Integer.toString(node.getKey());
            else
                return OPEN_BRACKET + ls + CLOSE_BRACKET;
        } else {
            String rs = postOrderStringKey(node.getChild(1), addInternalNodes);
            if (addInternalNodes)
                return OPEN_BRACKET + rs + CLOSE_BRACKET + Integer.toString(node.getKey());
            else
                return OPEN_BRACKET + rs + CLOSE_BRACKET;
        }
    }


    /**
     * Generate a string representing the traversing of the tree in post order. This method uses the
     * label values of the nodes.
     * 
     * @param n
     *        The root of the tree
     * @param addInternalNodes
     *        If <code>true</code>, the newick string includes the internal node labels. Otherwise,
     *        only leaf labels are present in the returned string.
     * @return A string representing the traversing of the tree in post order (using the label of
     *         the nodes).
     */
    private String postOrderStringLabel(TreeNode node, boolean addInternalNodes) {
        if (node.isLeaf())
            return node.getLabel();
        if (node.getNumberOfChildren() == 2) {
            String ls = postOrderStringLabel(node.getChild(0), addInternalNodes);
            String rs = postOrderStringLabel(node.getChild(1), addInternalNodes);
            if (addInternalNodes)
                return OPEN_BRACKET + ls + CHILD_SEPARATOR + rs + CLOSE_BRACKET + node.getLabel();
            else
                return OPEN_BRACKET + ls + CHILD_SEPARATOR + rs + CLOSE_BRACKET;
        } else if (node.getChild(0) != null) {
            String ls = postOrderStringLabel(node.getChild(0), addInternalNodes);
            if (addInternalNodes)
                return OPEN_BRACKET + ls + CLOSE_BRACKET + node.getLabel();
            else
                return OPEN_BRACKET + ls + CLOSE_BRACKET;
        } else {
            String rs = postOrderStringLabel(node.getChild(1), addInternalNodes);
            if (addInternalNodes)
                return OPEN_BRACKET + rs + CLOSE_BRACKET + node.getLabel();
            else
                return OPEN_BRACKET + rs + CLOSE_BRACKET;
        }
    }


    /**
     * Returns a String representation of this tree.
     * 
     * @return A String representation of this tree.
     */
    public String toString() {
        return postOrderStringKey(root, true);
    }


    /**
     * Returns a String representation of this tree.
     * 
     * @param label
     *        If <code>true</code>, returns a String that contains the labels of the nodes.
     *        Otherwise, returns a String that contains the key values of the nodes.
     * @return A String representation of this tree.
     */
    public String toString(boolean label) {
        if (label)
            return postOrderStringLabel(root, true);
        return postOrderStringKey(root, true);
    }


    /**
     * Returns a String representation of this tree.
     * 
     * @param label
     *        If <code>true</code>, returns a String that contains the labels of the nodes.
     *        Otherwise, returns a String that contains the key values of the nodes.
     * @param addInternalNodes
     *        If <code>true</code>, the newick string includes the internal node labels. Otherwise,
     *        only leaf labels are present in the returned string.
     * @return A String representation of this tree.
     */
    public String toString(boolean label, boolean addInternalNodes) {
        if (label)
            return postOrderStringLabel(root, addInternalNodes);
        return postOrderStringKey(root, addInternalNodes);
    }


    /**
     * Update the list of ancestor nodes of all nodes of the tree.
     */
    public void updateAncestorList() {
        updateAncestorList(this.root, new HashSet<TreeNode>());
    }


    /**
     * Recursive method to update the list of ancestors of all nodes of the tree.
     * 
     * @param node
     *        Node to be updated.
     * @param ancestors
     *        List of ancestors.
     */
    private void updateAncestorList(TreeNode node, HashSet<TreeNode> ancestors) {
        node.emptyAncestors();
        node.addAncestor(node);
        node.addAncestors(ancestors);
        TreeNode[] children = node.getChildren();
        if (children[0] != null) {
            updateAncestorList(children[0], node.getDescendants());
        }
        if (children[1] != null) {
            updateAncestorList(children[1], node.getDescendants());
        }
    }


    /**
     * Update the list of descendant nodes of all nodes of the tree.
     */
    public void updateDescendantList() {
        updateDescendantList(this.root);
    }


    /**
     * Recursive method to update the list of descendants of all nodes of the tree.
     * 
     * @param node
     *        Node to be updated.
     * @param ancestors
     *        List of ancestors.
     */
    private void updateDescendantList(TreeNode node) {
        node.emptyDescendants();
        node.addDescendant(node);
        TreeNode[] children = node.getChildren();
        if (children[0] != null) {
            updateDescendantList(children[0]);
            node.addDescendants(children[0].getDescendants());
        }
        if (children[1] != null) {
            updateDescendantList(children[1]);
            node.addDescendants(children[1].getDescendants());
        }
    }


    /**
     * Returns the list of descendants of the given tree node.
     * 
     * @param key
     *        Identifier of the tree node.
     * @return The list of descendants of the given tree node.
     */
    public HashSet<TreeNode> getDescendants(int key, boolean preprocessed) {

        TreeNode node = mapKeyToNode.get(key);
        if (node == null)
            throw new RuntimeException("Invalid key identifier");

        if (preprocessed)
            return node.getDescendants();

        return getDescendantsList(node);
    }


    /**
     * Returns the list of descendants of the given tree node.
     * 
     * @param node
     *        Root of the subtree.
     * @return The list of descendants of the given tree node.
     */
    private HashSet<TreeNode> getDescendantsList(TreeNode node) {
        HashSet<TreeNode> descendantsList = new HashSet<TreeNode>();
        if (node.isLeaf()) {
            descendantsList.add(node);
        } else {
            TreeNode child0 = node.getChild(0);
            TreeNode child1 = node.getChild(1);
            if (child0 != null)
                descendantsList.addAll(getDescendantsList(child0));
            if (child1 != null)
                descendantsList.addAll(getDescendantsList(child1));
            descendantsList.add(node);
        }
        return descendantsList;
    }


    /**
     * Returns the list of ancestors of the given tree node.
     * 
     * @param key
     *        Identifier of the tree node.
     * @return The list of descendants of the given tree node.
     */
    public HashSet<TreeNode> getAncestors(int key, boolean preprocessed) {
        TreeNode node = mapKeyToNode.get(key);
        if (node == null)
            throw new RuntimeException("Invalid key identifier");
        if (preprocessed)
            return node.getAncestors();
        return getAncestorsHashSet(node);
    }


    /**
     * Returns the list of descendants of the given tree node.
     * 
     * @param key
     *        Identifier of the tree node.
     * @return The list of descendants of the given tree node.
     */
    public ArrayList<TreeNode> getAncestorsArrayList(int key) {
        TreeNode node = mapKeyToNode.get(key);
        if (node == null)
            throw new RuntimeException("Invalid key identifier");
        ArrayList<TreeNode> ancestors = new ArrayList<TreeNode>();
        while (node != null) {
            ancestors.add(node);
            node = node.getParent();
        }
        return ancestors;
    }


    /**
     * Returns the list of descendants of the given tree node.
     * 
     * @param key
     *        Identifier of the tree node.
     * @return The list of descendants of the given tree node.
     */
    private HashSet<TreeNode> getAncestorsHashSet(TreeNode node) {
        HashSet<TreeNode> ancestors = new HashSet<TreeNode>();
        while (node != null) {
            ancestors.add(node);
            node = node.getParent();
        }
        return ancestors;
    }


    /**
     * Returns the table which contains the least common ancestor relationship for all pair of tree
     * nodes.
     * 
     * @return The table which contains the least common ancestor relationship for all pair of tree
     *         nodes.
     */
    private TreeNode[][] getLCATable() {
        if (lcaTable == null)
            createLCATable();
        return lcaTable;
    }


    /**
     * Returns the least common ancestor of the given pair of tree nodes.
     * 
     * @param u
     *        Node u
     * @param v
     *        Node v
     * @return The least common ancestor of u and v.
     */
    public TreeNode getLCA(int u, int v) {
        TreeNode nodeU = mapKeyToNode.get(u);
        TreeNode nodeV = mapKeyToNode.get(v);
        if (nodeU == null)
            throw new RuntimeException("Invalid key identifier for node u");
        if (nodeV == null)
            throw new RuntimeException("Invalid key identifier for node v");
        TreeNode[][] table = getLCATable();
        int indexU = nodes.indexOf(nodeU);
        int indexV = nodes.indexOf(nodeV);
        if (indexU == -1 || indexV == -1) {
            System.out.println(u + "\t" + indexU + "\t" + nodeU);
            System.out.println(v + "\t" + indexV + "\t" + nodeV);
            return null;
        }
        return table[indexU][indexV];
    }


    /**
     * Creates the LCA table of the tree. Notice that, only the nodes that are in the main tree are
     * included in the table.
     */
    public void createLCATable() {
        if (nodes == null)
            updateAttributes();
        lcaTable = new TreeNode[nodes.size()][];
        for (int i = 0; i < nodes.size(); i++) {
            lcaTable[i] = new TreeNode[nodes.size()];
            for (int j = 0; j < nodes.size(); j++) {
                lcaTable[i][j] = lca(nodes.get(i), nodes.get(j));
            }
        }
    }


    /**
     * Returns the LCA of the two given nodes. Warning 1: This method does not use the lca table and
     * performs the search for the LCA based on the height of the nodes. Warning 2: Only nodes which
     * are in the main tree (whose root node is assigned to the attribute root of this class) are
     * considered.
     * 
     * @param u
     *        Node u
     * @param v
     *        Node v
     * @return The LCA of the two given nodes.
     */
    private TreeNode lca(TreeNode u, TreeNode v) {

        if (!nodes.contains(u) || !nodes.contains(v))
            return null;

        /* First put the nodes into the same height. */
        while (u.getHeight() != v.getHeight()) {
            if (u.getHeight() > v.getHeight()) {
                u = u.getParent();
            } else {
                v = v.getParent();
            }
        }

        if (u == v)
            return u;

        /* Keep looking for the least common ancestor. */
        while (u.getParent().getKey() != v.getParent().getKey()) {
            u = u.getParent();
            v = v.getParent();
        }

        return u.getParent();
    }


    /**
     * Return an array containing all leaf nodes of the tree.
     * 
     * @return An array containing all leaf nodes of the tree.
     */
    public TreeNode[] getLeafNodes() {
        if (leaves == null)
            updateAttributes();
        return leaves.toArray(new TreeNode[0]);
    }


    /**
     * Return an array containing all leaf nodes of the tree.
     * 
     * @return An array containing all leaf nodes of the tree.
     */
    public TreeNode[] getNodes() {
        if (nodes == null)
            updateAttributes();
        return nodes.toArray(new TreeNode[0]);
    }


    /**
     * Creates a map between each partition of the tree and their number of occurrences.
     * 
     * @param leavesOrder
     *        An ordering of the leaves.
     * @return A map between each partition of the tree and their number of occurrences.
     */
    private HashMap<ArrayList<Integer>, Integer> partitionMap(HashMap<String, Integer> leavesOrder) {
        HashMap<ArrayList<Integer>, Integer> partitionMap = new HashMap<ArrayList<Integer>, Integer>();
        if (nodes == null)
            updateAttributes();
        for (TreeNode n: nodes) {
            ArrayList<Integer> p = n.partition(leavesOrder);
            if (partitionMap.containsKey(p)) {
                int i = partitionMap.get(p);
                partitionMap.put(p, i + 1);
            } else {
                partitionMap.put(n.partition(leavesOrder), 1);
            }
        }
        return partitionMap;
    }


    /**
     * Computes the (modifed)RF distance between this tree and the one given as argument using the
     * order specified in the map.
     * 
     * @param other
     *        Tree that will be compared to this tree.
     * @param leavesOrder
     *        An ordering of the leaves of this tree.
     * @return The (modifed)RF distance between this tree and the one given as argument.
     */
    public int mRFDistance(Tree other, HashMap<String, Integer> leavesOrder) {
        int distance = 0;
        HashMap<ArrayList<Integer>, Integer> partitionMap = other.partitionMap(leavesOrder);
        for (TreeNode n: nodes) {
            ArrayList<Integer> partition = n.partition(leavesOrder);
            if (partitionMap.containsKey(partition)) {
                int i = partitionMap.get(partition);
                if (i == 0) {
                    partitionMap.remove(partition);
                    distance++;
                } else {
                    partitionMap.put(partition, i - 1);
                }
            } else {
                distance++;
            }
        }
        for (int i: partitionMap.values()) {
            distance = distance + i;
        }
        return distance;
    }


    /**
     * Computes the MAAC value between this tree and the one given as argument.
     * 
     * @param other
     *        Tree that will be compared to this tree.
     * @return The MAAC value between this tree and the one given as argument.
     */
    public HashMultiset<String> computeMaac(Tree other) {

        if (nodes == null)
            updateAttributes();

        ArrayList<TreeNode> otherNodes = new ArrayList<TreeNode>(Arrays.asList(other.getNodes()));

        ArrayList<ArrayList<HashMultiset<String>>> maacTable = new ArrayList<ArrayList<HashMultiset<String>>>();
        for (int i = 0; i < nodes.size(); i++) {

            TreeNode v1 = nodes.get(i);
            maacTable.add(i, new ArrayList<HashMultiset<String>>(otherNodes.size()));

            for (int j = 0; j < otherNodes.size(); j++) {
                TreeNode v2 = otherNodes.get(j);

                if (v1.isLeaf() || v2.isLeaf()) {

                    HashMultiset<String> seti = HashMultiset.create();
                    HashMultiset<String> setj = HashMultiset.create();

                    for (TreeNode leaf: v1.getLeafSet())
                        seti.add(leaf.getLabel());

                    for (TreeNode leaf: v2.getLeafSet())
                        setj.add(leaf.getLabel());

                    HashMultiset<String> setij = HashMultiset.create(Multisets.intersection(seti,
                                                                                            setj));
                    maacTable.get(i).add(j, setij);

                } else {

                    HashMultiset<String> match = match(v1, v2, maacTable, nodes, otherNodes);
                    HashMultiset<String> diag = diag(v1, v2, maacTable, nodes, otherNodes);
                    if (match.size() > diag.size()) {
                        maacTable.get(i).add(j, match);
                    } else {
                        maacTable.get(i).add(j, diag);
                    }
                }
            }
        }

        return maacTable.get(nodes.size() - 1).get(otherNodes.size() - 1);
    }


    /**
     * Compute the MAAC distance between this tree and the one given as argument.
     * 
     * @param other
     *        Tree that will be compared to this tree.
     * @return The MAAC distance between this tree and the one given as argument.
     */
    public double computeMaacDistance(Tree other) {
        if (leaves == null)
            updateAttributes();
        double maacValue = computeMaac(other).size();
        double maxSize = Math.max(leaves.size(), other.getLeafNodes().length);
        return 1.0 - (maacValue / maxSize);
    }


    /**
     * Compute the MAAC similarity between this tree and the one given as argument.
     * 
     * @param other
     *        Tree that will be compared to this tree.
     * @return The MAAC similarity between this tree and the one given as argument.
     */
    public double computeMaacSimilarity(Tree other) {
        return 1 - computeMaacDistance(other);
    }


    /**
     * Auxiliar function for computing the MAAC between two trees.
     */
    private HashMultiset<String> match(TreeNode v1,
                                       TreeNode v2,
                                       ArrayList<ArrayList<HashMultiset<String>>> maacTable,
                                       ArrayList<TreeNode> po1,
                                       ArrayList<TreeNode> po2) {

        TreeNode c11 = v1.getChild(0);
        TreeNode c12 = v1.getChild(1);
        TreeNode c21 = v2.getChild(1);
        TreeNode c22 = v2.getChild(0);

        int oc11 = po1.indexOf(c11);
        int oc12 = po1.indexOf(c12);

        int oc21 = po2.indexOf(c21);
        int oc22 = po2.indexOf(c22);

        HashMultiset<String> a = maacTable.get(oc11).get(oc21);
        HashMultiset<String> c = maacTable.get(oc12).get(oc22);

        HashMultiset<String> b = maacTable.get(oc11).get(oc22);
        HashMultiset<String> d = maacTable.get(oc12).get(oc21);

        HashMultiset<String> match;
        if ((a.size() + c.size()) < (b.size() + d.size())) {
            match = HashMultiset.create(b.size() + d.size());
            match.addAll(b);
            match.addAll(d);
        } else {
            match = HashMultiset.create(a.size() + c.size());
            match.addAll(a);
            match.addAll(c);
        }

        return match;
    }


    /**
     * Auxiliar function for computing the MAAC between two trees.
     */
    private HashMultiset<String> diag(TreeNode v1,
                                      TreeNode v2,
                                      ArrayList<ArrayList<HashMultiset<String>>> maacTable,
                                      ArrayList<TreeNode> po1,
                                      ArrayList<TreeNode> po2) {

        int oc11 = po1.indexOf(v1.getChild(0));
        int oc12 = po1.indexOf(v1.getChild(1));
        int ov1 = po1.indexOf(v1);
        int oc21 = po2.indexOf(v2.getChild(0));
        int oc22 = po2.indexOf(v2.getChild(1));
        int ov2 = po2.indexOf(v2);
        HashMultiset<String> opt;

        HashMultiset<String> opt1 = maacTable.get(ov1).get(oc21);
        HashMultiset<String> opt2 = maacTable.get(ov1).get(oc22);
        HashMultiset<String> opt3 = maacTable.get(oc11).get(ov2);
        HashMultiset<String> opt4 = maacTable.get(oc12).get(ov2);

        if (opt1.size() < opt2.size())
            opt = HashMultiset.create(opt2);
        else
            opt = HashMultiset.create(opt1);

        if (opt.size() < opt3.size())
            opt = HashMultiset.create(opt3);

        if (opt.size() < opt4.size())
            opt = HashMultiset.create(opt4);

        return opt;
    }


    /**
     * Prints the LCA table of this tree.
     */
    public String printLCATable() {

        TreeNode[][] table = getLCATable();

        /* Prepare the header of the table */
        StringBuffer buffer = new StringBuffer();
        for (int i = 0; i < nodes.size(); i++)
            buffer.append("\t" + Integer.toString(nodes.get(i).getKey()));

        buffer.append("\n--");

        for (int i = 0; i < nodes.size(); i++)
            buffer.append("\t--");

        buffer.append("\n");

        /* Add one line for each node. */
        for (int i = 0; i < nodes.size(); i++) {
            buffer.append(Integer.toString(nodes.get(i).getKey()) + ":");

            for (int j = 0; j < nodes.size(); j++)
                buffer.append("\t" + Integer.toString(table[i][j].getKey()));

            buffer.append("\n");
        }

        return buffer.toString();

    }


    /**
     * Returns the mapping between labels and nodes. WARNING!!! This method assumes that the labels
     * are unique.
     * 
     * @return The mapping between labels and nodes.
     */
    public HashMap<String, TreeNode> getMapLabelToNode() {
        if (mapLabelToNode == null) {

            if (nodes == null)
                updateAttributes();

            mapLabelToNode = new HashMap<String, TreeNode>();
            for (TreeNode node: nodes) {
                if (mapLabelToNode.containsKey(node.getLabel()))
                    throw new RuntimeException("This method should not be applied for tree which have non-unique labels.");
                mapLabelToNode.put(node.getLabel(), node);
            }
        }
        return mapLabelToNode;
    }


    /**
     * Returns all nodes that are in the path from the descendant to the ancestor.
     * 
     * @param descendant
     *        Descendant node.
     * @param ancestor
     *        Ancestor node.
     * @return All nodes that are in the path from the descendant to the ancestor.
     */
    public ArrayList<TreeNode> getPathFromDescendantToAncestor(int descendant, int ancestor) {

        TreeNode descendantNode = mapKeyToNode.get(descendant);
        TreeNode ancestorNode = mapKeyToNode.get(ancestor);

        ArrayList<TreeNode> toReturn = new ArrayList<TreeNode>();
        while (descendantNode != null && descendantNode != ancestorNode) {
            toReturn.add(descendantNode);
            descendantNode = descendantNode.getParent();
        }
        toReturn.add(ancestorNode);

        if (descendantNode == null)
            return null;

        return toReturn;
    }

}
