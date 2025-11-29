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

package threads;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.Random;

import generator.IParasiteGenerator;
import generator.IScenario;
import generator.coalescencemodel.CoalescenceModel;
import generator.defaultmodel.DefaultModel;
import trees.Tree;
import util.Statistics;

public class QuantileDistribution extends GenerateDistribution {

    /** Auxiliary constant for the perturbation procedure. */
    private static final double DENOMINATOR = 1000000.0;

    /** List of vectors from where we sample vectors that are going to be perturbated. */
    private ArrayList<double[]> samplingPopulation;

    /** Epsilon to accept new vectors. */
    private double epsilon;

    /** Left bound for the permutation procedure */
    private double transformationLeft;

    /** Right bound for the permutation procedure */
    private int transformationRight;

    /** Random generator. */
    private Random random;

    /**
     * Maximum number of rejected vectors (If this number is reached, the process is aborted.)
     */
    private int maximumNumberOfRejectedVectors;


    /**
     * Thread to generate vectors for the prior distribution.
     * 
     * @param vectorManager
     *        Object that manage the access to the list of vectors in a synchronized way.
     * @param samplingPopulation
     *        ArrayList that contain the vectors that are going to be sampled.
     * @param numberOfVectors
     *        Number of vectors to generate.
     * @param numberOfTrees
     *        Number of trees that must be generated for each vector.
     * @param maximumNumberOfRejectedVectors
     *        Maximum number of rejected vectors (If this number is reached, the process is
     *        aborted.)
     * @param hostTree
     *        Host tree.
     * @param parasiteTree
     *        Parasite tree.
     * @param mappingParasiteHost
     *        Mapping of the parasite leaf nodes into the host leaf nodes.
     * @param model
     *        Model to be used during the simulation.
     * @param metric
     *        Metric to be used during the simulation.
     * @param maximumNumberOfTreesFactor
     *        Maximum number of trees factor.
     * @param cyclicityTestModel
     *        Cyclicity test model identifier.
     * @param alpha1
     *        Alpha 1 value for computing the MMAAC distance.
     * @param alpha2
     *        Alpha 2 value for computing the MMAAC distance.
     * @param rootMappingProbability
     *        Probability of mapping into the root node.
     * @param pertubation
     *        Maximum value which can be added/subtracted from the probability value in the
     *        perturbation procedure.
     * @param epsilon
     *        Epsilon used to select elements for this quantile.
     */
    public QuantileDistribution(SynchronizedVectorManager vectorManager,
                                ArrayList<double[]> samplingPopulation,
                                int numberOfVectors,
                                int numberOfTrees,
                                int maximumNumberOfRejectedVectors,
                                Tree hostTree,
                                Tree parasiteTree,
                                HashMap<String, String> mappingParasiteHost,
                                int model,
                                int metric,
                                int maximumNumberOfTreesFactor,
                                int cyclicityTestModel,
                                double alpha1,
                                double alpha2,
                                double rootMappingProbability,
                                double perturbation,
                                double epsilon) {
        super(vectorManager,
              numberOfVectors,
              numberOfTrees,
              hostTree,
              parasiteTree,
              mappingParasiteHost,
              model,
              metric,
              maximumNumberOfTreesFactor,
              cyclicityTestModel,
              alpha1,
              alpha2,
              rootMappingProbability);

        this.maximumNumberOfRejectedVectors = maximumNumberOfRejectedVectors;
        this.samplingPopulation = samplingPopulation;
        this.epsilon = epsilon;

        this.transformationLeft = perturbation * DENOMINATOR;
        this.transformationRight = (int)(2 * transformationLeft) + 1;

        this.random = new Random();

    }


    public void run() {
        if (model == GenerateDistribution.DEFAULT_MODEL) {
            runDefaultModel();
        } else {
            runCoalescentModel();
        }
    }


    private void runDefaultModel() {

        NumberFormat format = NumberFormat.getInstance();
        format = NumberFormat.getInstance(Locale.US);
        format.setMinimumFractionDigits(2);
        format.setMaximumFractionDigits(2);

        while (vectorManager.getNumberOfVectors() < numberOfVectors
               && vectorManager.getNumberOfRejectedVectors() < maximumNumberOfRejectedVectors) {

            double[] vector = samplingPopulation.get(random.nextInt(samplingPopulation.size()));
            double[] newvector = transformVector(vector);

            updateExpectedNumberOfEvents(newvector[0], newvector[1], newvector[2], newvector[3]);

            IParasiteGenerator generator = new DefaultModel(hostTree,
                                                            newvector[0],
                                                            newvector[1],
                                                            newvector[2],
                                                            parasiteTree.getNumberOfLeafNodes(),
                                                            maximumNumberOfLeaves,
                                                            cyclicityTestModel,
                                                            MAXIMUM_SIZE_STOP_CRITERIUM,
                                                            rootMappingProbability);

            int nScenarios = 0;
            double[] values = new double[numberOfTrees];
            for (int j = 0; nScenarios < numberOfTrees && j < maximumNumberOfTrees; j++) {
                IScenario scenario = generator.generateParasiteTree();
                if (scenario != null) {
                    double distance = 1.0;
                    switch (metric) {
                        case METRIC_MAAC_DISTANCE:
                            distance = computeMAACDistanceFromRealParasite(scenario);
                            break;
                        case METRIC_LEAVES_AND_MAAC_DISTANCE:
                            distance = computeLEAVES_AND_MAAC_Metric(scenario);
                            break;
                        case METRIC_EVENTS_AND_MAAC_DISTANCE:
                            distance = computeEVENTS_AND_MAAC_Metric(scenario);
                            break;
                    }
                    values[nScenarios++] = distance;
                }
            }

            double representativeDistance = 1.0;
            if (nScenarios == numberOfTrees) {
                representativeDistance = Statistics.mean(values);
            }

            if (representativeDistance <= epsilon) {
                double[] array = {newvector[0], newvector[1], newvector[2], newvector[3],
                                  representativeDistance};
                vectorManager.registerVector(array);
            } else {
                vectorManager.increaseNumberOfRejectedVectors();
            }

        }

    }


    private void runCoalescentModel() {

        while (vectorManager.getNumberOfVectors() < numberOfVectors
               && vectorManager.getNumberOfRejectedVectors() < maximumNumberOfRejectedVectors) {

            double[] vector = samplingPopulation.get(random.nextInt(samplingPopulation.size()));
            double[] newvector = transformVector(vector);

            IParasiteGenerator generator = new CoalescenceModel(hostTree,
                                                                parasiteTree,
                                                                mappingParasiteHost,
                                                                newvector[0],
                                                                newvector[1],
                                                                true,
                                                                model == COALESCENCE_WITH_DISTANCE,
                                                                cyclicityTestModel);

            double[] distances = new double[numberOfTrees];
            double[] frequencies = new double[numberOfTrees];
            int nScenarios = 0;
            for (int j = 0; nScenarios < numberOfTrees && j < maximumNumberOfTrees; j++) {
                IScenario scenario = generator.generateParasiteTree();
                if (scenario != null) {
                    double distance = 1.0;
                    switch (metric) {
                        case METRIC_MAAC_DISTANCE:
                            distance = computeMAACDistanceFromRealParasite(scenario);
                            break;
                        case METRIC_LEAVES_AND_MAAC_DISTANCE:
                            distance = computeLEAVES_AND_MAAC_Metric(scenario);
                            break;
                        case METRIC_EVENTS_AND_MAAC_DISTANCE:
                            distance = computeEVENTS_AND_MAAC_Metric(scenario);
                            break;
                    }
                    distances[nScenarios] = distance;
                    frequencies[nScenarios] = scenario.getFrequencyOfLosses();
                    nScenarios++;
                }
            }

            double representativeDistance = 1.0;
            double representativeFrequency = 1.0;
            if (nScenarios == maximumNumberOfTrees) {
                representativeDistance = Statistics.mean(distances);
                representativeFrequency = Statistics.mean(frequencies);
            }

            /* Compute the representative values. */
            if (representativeDistance <= epsilon) {
                double[] array = {newvector[0], newvector[1], newvector[2],
                                  representativeFrequency, representativeDistance};
                vectorManager.registerVector(array);
            } else {
                vectorManager.increaseNumberOfRejectedVectors();
            }

        }

    }


    /**
     * Transform the given vector.
     * 
     * @param vector
     *        Vector to be transformed.
     * @return The transformed vector.
     */
    private double[] transformVector(double[] vector) {

        int length = 4;
        if (model != DEFAULT_MODEL) {
            length = 3;
        }

        double sum = 0.0;
        double[] newvector = new double[length];

        for (int i = 0; i < length; i++) {
            double n = (transformationLeft - random.nextInt(transformationRight)) / DENOMINATOR;
            newvector[i] = vector[i] - n;
            if (newvector[i] < 0) {
                newvector[i] = 0;
            }
            sum += newvector[i];
        }

        for (int i = 0; i < length; i++) {
            newvector[i] = newvector[i] / sum;
        }

        return newvector;

    }

}
