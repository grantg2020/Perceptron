package com.grantgupton.ml.perceptron;

import java.util.Random;

public class MultiLayerPerceptron extends Perceptron {

	/** Size of each layer in network */
	private int[] layerSizes;

	/** List of layers */
	private Layer[] layers;

	/**
	 * Default constructor
	 *
	 * @param layers       array of layer sizes
	 * @param learningRate learning rate
	 */
	public MultiLayerPerceptron(int[] layers, double learningRate) {
		super(learningRate);

		layerSizes = layers;
		initLayers(layerSizes);
	}

	/**
	 * Constructor with default learning rate
	 *
	 * @param layers array of layer sizes
	 */
	public MultiLayerPerceptron(int[] layers) {
		this(layers, DEFAULT_LEARNING_RATE);
	}

	/**
	 * Training with infinite epochs until within minimum error
	 * 
	 * @param data data to train
	 * @return number of epochs spent
	 */
	public double train(PerceptronData[] data) {
		return train(data, INFINITE_EPOCHS);
	}

	@Override
	public double train(PerceptronData data) {
		// TODO Auto-generated method stub

		return 0;
	}

	@Override
	public double[] predict(double[] inputs) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int[] size() {
		return layerSizes;
	}

	/**
	 * Returns the sigma function results (1/(1 + e^-x))
	 * 
	 * @param value value to calculate
	 * @return result of calculation
	 */
	private double sigma(double value) {
		return 1.0 / (1.0 + Math.pow(Math.E, -value));
	}

	/**
	 * Returns the sigma prime function results sigma * (1 - sigma)
	 * 
	 * @param value value to calculate
	 * @return result of calculation
	 */
	private double sigmaPrime(double value) {
		return sigma(value) * (1.0 - sigma(value));
	}

	@Override
	public double activation(double value) {
		return sigmaPrime(value);
	}

	@Override
	public String toString() {
		String str = "";
		for (Layer l : layers) {
			str += l.toString();
		}
		return str;
	}

	/**
	 * Inits all the layers
	 * 
	 * @param layerSizes sizes of each layer
	 */
	private void initLayers(int[] layerSizes) {
		layers = new Layer[layerSizes.length];

		for (int i = 0; i < layerSizes.length; i++) {
			Layer l = new Layer(layerSizes[i], i < layerSizes.length - 1 ? layerSizes[i + 1] : 0);
			layers[i] = l;
		}
	}

	private class Layer {
		/** Size of layer */
		int size;

		/** Neurons in current layer */
		Neuron[] neurons;

		/**
		 * Default constructor
		 * 
		 * @param size size of layer
		 */
		public Layer(int size, int nextLayerSize) {
			neurons = new Neuron[size];

			this.size = size;
			for (int i = 0; i < size; i++) {
				neurons[i] = new Neuron(nextLayerSize);
			}
		}

		/**
		 * Returns size of layer
		 * 
		 * @return size of layer
		 */
		public int size() {
			return size;
		}

		@Override
		public String toString() {
			String str = "";
			for (Neuron n : neurons) {
				str += n.getWeights().length;
			}
			return str;
		}
	}

	private class Neuron {
		/** Weights from this neuron to next layer neurons */
		private double[] weights;

		/** Value of neuron */
		private double value;

		/** Bias of neuron */
		private double bias;

		/**
		 * Default constructor
		 */
		private Neuron(int nextLayerSize) {
			Random rand = new Random();
			double[] w = new double[nextLayerSize];

			// Fill with random starting weights
			for (int i = 0; i < nextLayerSize; i++) {
				w[i] = rand.nextDouble();
			}
			setValue(0);
			setBias(rand.nextDouble());
		}

		/**
		 * Returns value
		 * 
		 * @return value
		 */
		public double getValue() {
			return value;
		}

		/**
		 * Sets the value
		 * 
		 * @param value value to set
		 */
		public void setValue(double value) {
			this.value = value;
		}

		/**
		 * Sets the bias
		 * 
		 * @param bias bias to set
		 */
		public void setBias(double value) {
			this.bias = value;
		}

		/**
		 * Returns bias
		 * 
		 * @return bias
		 */
		public double getBias() {
			return bias;
		}

		/**
		 * Returns weights
		 * 
		 * @return weights
		 */
		public double[] getWeights() {
			return weights;
		}

		/**
		 * Sets weights
		 * 
		 * @param weight weight to sets
		 */
		public void setWeights(double[] weight) {
			this.weights = weight;
		}

		/**
		 * Changes weight by some amount
		 * 
		 * @param weight weight to add
		 */
		public void changeWeight(double weight, int index) {
			this.weights[index] += weight;
		}
	}
}
