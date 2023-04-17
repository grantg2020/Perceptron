package com.grantgupton.ml.perceptron;

public class PerceptronData {
	/** Input data */
	private double[] inputs;

	/** Output data */
	private double[] outputs;

	/**
	 * Default constructor
	 * 
	 * @param inputs  input data
	 * @param outputs output data
	 */
	public PerceptronData(double[] inputs, double[] outputs) {
		setInputs(inputs);
		setOutputs(outputs);
	}

	/**
	 * Returns inputs
	 * 
	 * @return inputs
	 */
	public double[] getInputs() {
		return inputs;
	}

	/**
	 * Sets inputs
	 * 
	 * @param inputs inputs to set
	 */
	public void setInputs(double[] inputs) {
		this.inputs = inputs;
	}

	/**
	 * Returns outputs
	 * 
	 * @return outputs
	 */
	public double[] getOutputs() {
		return outputs;
	}

	/**
	 * Sets the outputs
	 * 
	 * @param outputs outputs to set
	 */
	public void setOutputs(double[] outputs) {
		this.outputs = outputs;
	}

}
