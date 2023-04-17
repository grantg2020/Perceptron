package com.grantgupton.ml.perceptron;

public abstract class Perceptron {
	/** Default learning rate */
	public static double DEFAULT_LEARNING_RATE = 0.01;

	/** Value for infinite training */
	public static int INFINITE_EPOCHS = -1;

	/** Maximum error */
	public static double MAX_ERROR = 0.01;

	/** Learning rate for training */
	private double learningRate;

	/** Total epochs run */
	private int epochs;

	public Perceptron(double learningRate) {
		setLearningRate(learningRate);
		epochs = 0;
	}

	/**
	 * Returns size array of network
	 * 
	 * @return size array of network
	 */
	public abstract int[] size();

	/**
	 * Trains network with array of data
	 * 
	 * @param data      array of data to train with
	 * @param maxEpochs maximum epochs before stopping training
	 * @return number of epochs spent
	 */
	public int train(PerceptronData[] data, int maxEpochs) {
		double error;
		do {
			error = 0;
			for (PerceptronData d : data) {
				error += Math.abs(train(d));
			}
			epochs++;
			if (epochs % 100 == 0)
				System.out.println("Error: " + error + "; Epoch: " + epochs);
		} while (error > MAX_ERROR && (maxEpochs < 0 || epochs < maxEpochs));

		return epochs;
	}

	/**
	 * Training helper for only one set of data
	 * 
	 * @param data data to train with
	 * @return error
	 */
	public abstract double train(PerceptronData data);

	/**
	 * Predicts outputs based on input array
	 * 
	 * @param inputs array of inputs
	 * @return output prediction
	 */
	public abstract double[] predict(double[] inputs);

	/**
	 * Activation method to normalize a value
	 * 
	 * @param value value to normalize
	 * @return normalized value
	 */
	public abstract double activation(double value);

	/**
	 * Returns learning rate
	 * 
	 * @return learning rate
	 */
	public double getLearningRate() {
		return learningRate;
	}

	/**
	 * Sets the learning rate
	 * 
	 * @param learningRate the learningRate to set
	 */
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	/**
	 * @return the epochs
	 */
	public int getEpochs() {
		return epochs;
	}
}
