package com.grantgupton.ml.perceptron;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Tests MultiLayer Perceptron
 * 
 * @author grantgupton
 *
 */
class MultiLayerPerceptronTest {
	/** MLP object */
	MultiLayerPerceptron mlpXOR;

	/** Training data */
	PerceptronData[] trainData = { new PerceptronData(new double[] { 1, 0 }, new double[] { 1 }),
			new PerceptronData(new double[] { 1, 1 }, new double[] { 0 }),
			new PerceptronData(new double[] { 0, 1 }, new double[] { 1 }),
			new PerceptronData(new double[] { 0, 0 }, new double[] { 0 }), };

	/**
	 * @throws java.lang.Exception
	 */
	@BeforeEach
	void setUp() throws Exception {
		mlpXOR = new MultiLayerPerceptron(new int[] { 2, 3, 1 });
	}

	@Test
	void testGetLearningRate() {
		assertEquals(Perceptron.DEFAULT_LEARNING_RATE, mlpXOR.getLearningRate());
	}

	@Test
	void testSetLearningRate() {
		double newLR = 0.003434;
		mlpXOR.setLearningRate(newLR);
		assertEquals(newLR, mlpXOR.getLearningRate());
	}

	@Test
	void testTrainAndPredict() {
		mlpXOR.train(trainData);
		System.out.println(mlpXOR);
		assertEquals(new double[] { 0 }, mlpXOR.predict(new double[] { 1, 1 }));
		assertEquals(new double[] { 1 }, mlpXOR.predict(new double[] { 0, 1 }));
		assertEquals(new double[] { 0 }, mlpXOR.predict(new double[] { 0, 0 }));
		assertEquals(new double[] { 1 }, mlpXOR.predict(new double[] { 1, 0 }));
	}

}
