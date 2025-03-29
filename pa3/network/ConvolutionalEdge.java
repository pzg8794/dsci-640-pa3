/**
 * This class represents an ConvolutionalEdge in a neural network. It will contain
 * the ConvolutionalEdge's weight, and also have references to input node and output
 * nodes of this edge.
 */
package network;

import java.util.Random;
import util.Log;

public class ConvolutionalEdge extends Edge {
    //the weight for this edge
    public double weight[][][];

    //the delta calculated by backpropagation for this edge
    public double weightDelta[][][];

    /**
     * This constructs a new edge in the neural network between the passed
     * parameters. It will register itself at the input and output nodes
     * through the Node.addOutgoingConvolutionalEdge(ConvolutionalEdge) and Node.addIncomingConvolutionalEdge(ConvolutionalEdge)
     * methods.
     *
     * @param inputNode is the input for this edge
     * @param outputNode is the output for this edge
     */
    public ConvolutionalEdge(ConvolutionalNode inputNode, ConvolutionalNode outputNode, int sizeZ, int sizeY, int sizeX) throws NeuralNetworkException {
        super(inputNode, outputNode, sizeZ, sizeY, sizeX);
        this.inputNode = inputNode;
        this.outputNode = outputNode;

        if (inputNode.sizeZ - sizeZ + 1 != outputNode.sizeZ
                || inputNode.sizeY - sizeY + 1 != outputNode.sizeY - (2 * outputNode.padding)
                || inputNode.sizeX - sizeX + 1 != outputNode.sizeX - (2 * outputNode.padding)) {
            throw new NeuralNetworkException("Cannot connect input node " + inputNode.toString() + " to output node " + outputNode.toString() + " because sizes do not work with this filter (" + sizeZ + "x" + sizeY + "x" + sizeX  + "), output node size should be (batchSize x" + (inputNode.sizeZ - sizeZ + 1) + "x" + (inputNode.sizeY - sizeY + 1) + "x" + (inputNode.sizeX - sizeX + 1) + ")");
        }

        //initialize the weight and delta to 0
        weight = new double[sizeZ][sizeY][sizeX];
        weightDelta = new double[sizeZ][sizeY][sizeX];
    }

    /**
     * Resets the deltas for this edge
     */
    public void reset() {
        //Log.info("resetting convolutional edge with sizeZ: " + sizeZ + ", sizeY: " + sizeY + ", sizeX: " + sizeX);
        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weightDelta[z][y][x] = 0;
                }
            }
        }
    }

    /**
     * Used to get the weights of this Edge.
     * It will set the weights in the weights
     * parameter passed in starting at position, and return the number of
     * weights it set.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weights is the array of weights we're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeights(int position, double[] weights) {
        int weightCount = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weights[position + weightCount] = weight[z][y][x];
                    weightCount++;
                }
            }
        }

        return weightCount;
    }

    /**
     * Used to print gradients related to this edge, along with informationa
     * about this edge.
     * It start printing the gradients passed in starting at position, and 
     * return the number of gradients it printed.
     *
     * @param position is the index to start printing different gradients
     * @param numericGradient is the array of the numeric gradient we're printing
     * @param backpropGradient is the array of the backprop gradient we're printing
     *
     * @return the number of gradients printed by this edge
     */
    public int printGradients(int position, double[] numericGradient, double[] backpropGradient) {
        //don't print anything out, but print out this edge
        Log.info("ConvolutionalEdge from Node [layer: " + inputNode.layer + ", number: " + inputNode.number + "] to Node [layer: " + outputNode.layer + ", number: " + outputNode.number + "] to Node:");

        int count = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    Log.info("\tweights[" + z + "][" + y + "][" + x + "]: "+ Log.twoGradients(numericGradient[position + count], backpropGradient[position + count]));
                    count++;
                }
            }
        }

        return count;
    }


    /**
     * Used to get the deltas of this Edge.
     * It will set the deltas in the deltas
     * parameter passed in starting at position, and return the number of
     * deltas it set.
     *
     * @param position is the index to start setting deltas in the deltas parameter
     * @param deltas is the array of deltas we're setting.
     *
     * @return the number of deltas set in the deltas parameter
     */
    public int getDeltas(int position, double[] deltas) {
        int deltaCount = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    deltas[position + deltaCount] = weightDelta[z][y][x];
                    deltaCount++;
                }
            }
        }

        return deltaCount;
    }


    /**
     * Used to set the weights of this Edge.
     * It uses the same technique as Node.getWeights
     * where the starting position of weights to set is passed, and it returns
     * how many weights were set.
     * 
     * @param position is the starting position in the weights parameter to start
     * setting weights from.
     * @param weights is the array of weights we are setting from
     *
     * @return the number of weights gotten from the weights parameter
     */
    public int setWeights(int position, double[] weights) {
        int weightCount = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weight[z][y][x] = weights[position + weightCount];
                    weightCount++;
                }
            }
        }

        return weightCount;
    }



    /**
     * This initializes the weights of this ConvolutionalEdge (Filter) by
     * the range calculated by it's output node (which should be sqrt(2)/sqrt(all incoming edge filter sizes).
     *
     * @param range is sqrt(2)/sqrt(sum of output node incoming filter sizes)
     */
    public void initializeKaiming(double range, int fanIn) {
        //DONE: Implement this for Programming Assignment 3 - Part 1
        Random rand = new Random();

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    // Gaussian distribution: mean = 0, stddev = range
                    weight[z][y][x] = rand.nextGaussian() * range;
                }
            }
        }
    }

    /**
     * This initializes the weights of this ConvolutionalEdge (Filter) by
     * uniformly within the range calculated by it's output node (which 
     * should be between negative and positive sqrt(6)/sqrt(all incoming 
     * and outgoing edge filter sizes).
     *
     * @param range is sqrt(6)/sqrt(sum of output node incoming and outgoing filter sizes)
     */
    public void initializeXavier(double range, int fanIn, int fanOut) {
        //DONE: Implement this for Programming Assignment 3 - Part 1
        Random rand = new Random();

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    // Uniform distribution: [-range, +range]
                    weight[z][y][x] = (rand.nextDouble() * 2 * range) - range;
                }
            }
        }
    }


    /**
     * This takes an incoming delta from the output node
     * and propagates it backwards to the input node.
     *
     * @param delta is the delta/error from the output node.
     */
    public void propagateForward(double[][][][] inputValues) {
        //DONE: You need to implement this for Programming Assignment 3 - Part 1
        
        // Extract padding for clarity
        int padding = outputNode.padding;

        // Iterate through each image in the batch
        for(int imgIndex = 0; imgIndex < outputNode.inputValues.length; imgIndex++) {
            processImage(inputValues, imgIndex, padding);
        }
    }

    /**
     * Processes an individual image by iterating over its channels and applying convolution.
     * 
     * @param inputValues The input values from the previous layer.
     * @param imgIndex Index of the image in the batch.
     * @param padding The padding size for the convolution operation.
     */
    private void processImage(double[][][][] inputValues, int imgIndex, int padding) {
        for(int chIndex = 0; chIndex < outputNode.inputValues[imgIndex].length; chIndex++) {
            processChannel(inputValues, imgIndex, chIndex, padding);
        }
    }

    /**
     * Processes a single channel of an image, performing convolution over rows and columns.
     * 
     * @param inputValues The input values from the previous layer.
     * @param imgIndex Index of the image in the batch.
     * @param chIndex Index of the channel being processed.
     * @param padding The padding size for the convolution operation.
     */
    private void processChannel(double[][][][] inputValues, int imgIndex, int chIndex, int padding) {
        int numRows = outputNode.inputValues[imgIndex][chIndex].length - 2 * padding;
        int numCols = outputNode.inputValues[imgIndex][chIndex][0].length - 2 * padding;

        for(int col = 0; col < numRows; col++) {
            for(int row = 0; row < numCols; row++) {
                applyFilter(inputValues, imgIndex, chIndex, col, row, padding);
            }
        }
    }

    /**
     * Applies the filter to a specific region of the image and accumulates the result.
     * 
     * @param inputValues The input values from the previous layer.
     * @param imgIndex Index of the image in the batch.
     * @param chIndex Index of the channel being processed.
     * @param col Current column index of the convolution operation.
     * @param row Current row index of the convolution operation.
     * @param padding The padding size for the convolution operation.
     */
    private void applyFilter(double[][][][] inputValues, int imgIndex, int chIndex, int col, int row, int padding) {
        for(int filterCh = 0; filterCh < weight.length; filterCh++) {
            for(int filterCol = 0; filterCol < weight[filterCh].length; filterCol++) {
                for(int filterRow = 0; filterRow < weight[filterCh][filterCol].length; filterRow++) {
                    // Apply convolution operation and accumulate the result
                    outputNode.inputValues[imgIndex][chIndex][col + padding][row + padding] += 
                        weight[filterCh][filterCol][filterRow] * 
                        inputValues[imgIndex][chIndex + filterCh][col + filterCol][row + filterRow];
                }
            }
        }
    }



    /**
     * This takes an incoming delta from the output node
     * and propagates it backwards to the input node.
     *
     * @param delta is the delta/error from the output node.
     */
    public void propagateBackward(double[][][][] delta) {
        //TODO: You need to implement this for Programming Assignment 3 - Part 2

    }

}
