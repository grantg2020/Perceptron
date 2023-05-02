package com.grantgupton.ml.perceptron;

import java.util.Random;

public class SingleLayerPerceptron extends Perceptron {
	/** Number of inputs */
	private final int numInputs;

	/** Number of outputs */
	private final int numOutputs;

	/** List of input neurons */
	private Neuron[] neurons;

	/** Bias for output neuron */
	private double bias;

	/**
	 * Default constructor
	 * 
	 * @param inputs       number of inputs
	 * @param learningRate learning rate for training
	 */
	public SingleLayerPerceptron(int inputs, double learningRate) {
		super(learningRate);
		numInputs = inputs;
		numOutputs = 1; // Only one output for SLP
		bias = new Random().nextDouble();

		neurons = new Neuron[inputs];

		for (int i = 0; i < neurons.length; i++) {
			neurons[i] = new Neuron();
		}
	}

	/**
	 * Constructor with default learning rate
	 * 
	 * @param inputs number of inputs
	 */
	public SingleLayerPerceptron(int inputs) {
		this(inputs, DEFAULT_LEARNING_RATE);
	}

	@Override
	public int[] size() {
		return new int[] { numInputs, numOutputs };
	}

	@Override
	public double train(PerceptronData data) {
		if (data.getInputs().length != numInputs)
			throw new IllegalArgumentException("Invalid inputs");
		if (data.getOutputs().length != numOutputs)
			throw new IllegalArgumentException("Invalid outputs");

		double[] outputs = predict(data.getInputs());
		double error = data.getOutputs()[0] - outputs[0];

		for (int i = 0; i < outputs.length; i++) {
			neurons[i].changeWeight(error * data.getInputs()[i] * getLearningRate());
		}

		changeBias(error * getLearningRate());
		return error;
	}

	@Override
	public double[] predict(double[] inputs) {
		if (inputs.length != numInputs)
			throw new IllegalArgumentException("Invalid inputs");

		double[] out = new double[numOutputs];

		for (int i = 0; i < inputs.length; i++) {
			neurons[i].setValue(inputs[i]);
		}

		double sum = 0;
		for (Neuron n : neurons) {
			sum += n.getWeight() * n.getValue();
		}

		out[0] = activation(sum + getBias());

		return out;
	}

	/**
	 * Returns bias for output neuron
	 * 
	 * @return bias for output neuron
	 */
	public double getBias() {
		return bias;
	}

	/**
	 * Sets the bias
	 * 
	 * @param bias bias to set
	 */
	public void setBias(double bias) {
		this.bias = bias;
	}

	/**
	 * Changes bias value by amount
	 * 
	 * @param amount amount to change bias by
	 */
	public void changeBias(double amount) {
		this.bias += amount;
	}

	@Override
	public double activation(double value) {
		return value > 0 ? 1 : 0;
	}

	@Override
	public String toString() {
		String str = "";
		for (Neuron n : neurons) {
			str += "Neuron: " + n.getValue() + ", w: " + n.getWeight() + "; ";
		}
		str += "\nEpochs: " + getEpochs();
		return str;
	}

	private class Neuron {
		/** Weight from this neuron to output neuron */
		private double weight;

		/** Value of neuron */
		private double value;

		/**
		 * Default constructor
		 */
		private Neuron() {
			setWeight(new Random().nextDouble());
			setValue(0);
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
		 * Returns weight
		 * 
		 * @return weight
		 */
		public double getWeight() {
			return weight;
		}

		/**
		 * Sets weight
		 * 
		 * @param weight weight to set
		 */
		public void setWeight(double weight) {
			this.weight = weight;
		}

		/**
		 * Changes weight by some amount
		 * 
		 * @param weight weight to add
		 */
		public void changeWeight(double weight) {
			this.weight += weight;
		}
	}

}
