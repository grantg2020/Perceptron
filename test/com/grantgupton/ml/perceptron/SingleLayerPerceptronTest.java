package com.grantgupton.ml.perceptron;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Test class for single layer perceptron
 */
public class SingleLayerPerceptronTest {
	SingleLayerPerceptron slp;

	// Linearly separable dataset
	PerceptronData[] trainingSet = { new PerceptronData(new double[] { 0, 0 }, new double[] { 1 }),
			new PerceptronData(new double[] { 1, 1 }, new double[] { 0 }),
			new PerceptronData(new double[] { 0.4, 0.4 }, new double[] { 1 }),
			new PerceptronData(new double[] { 0.6, 0.6 }, new double[] { 0 }),
			new PerceptronData(new double[] { 0.45, 0.45 }, new double[] { 1 }),
			new PerceptronData(new double[] { 0.55, 0.55 }, new double[] { 0 }),
			new PerceptronData(new double[] { 0.1, 0.1 }, new double[] { 1 }),
			new PerceptronData(new double[] { 0.95, 0.55 }, new double[] { 0 }),
			new PerceptronData(new double[] { 0.55, 0.95 }, new double[] { 0 }), };

	@BeforeEach
	void setUp() throws Exception {
		slp = new SingleLayerPerceptron(2);
	}

	@Test
	void testTrainAndPredict() {
		slp.train(trainingSet, Perceptron.INFINITE_EPOCHS);

		System.out.println("Bias: " + slp.getBias());
		System.out.println(slp);

		assertEquals(1.0, slp.predict(new double[] { 0.1, 0.1 })[0], 0.0001);
		assertEquals(1.0, slp.predict(new double[] { 0.3, 0.3 })[0], 0.0001);
		assertEquals(1.0, slp.predict(new double[] { 0.2, 0.4 })[0], 0.0001);
		assertEquals(0.0, slp.predict(new double[] { 0.9, 1 })[0], 0.0001);
		assertEquals(1.0, slp.predict(new double[] { 0.44, 0.44 })[0], 0.0001);
		assertEquals(0.0, slp.predict(new double[] { 0.56, 0.56 })[0], 0.0001);
	}

	@Test
	void testGetLearningRate() {
		assertEquals(Perceptron.DEFAULT_LEARNING_RATE, slp.getLearningRate());
		slp.setLearningRate(0.1);
		assertEquals(0.1, slp.getLearningRate());
	}

}
