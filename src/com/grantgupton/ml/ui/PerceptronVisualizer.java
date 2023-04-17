package com.grantgupton.ml.ui;

import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.image.BufferedImage;
import java.util.function.Consumer;

import javax.swing.JFrame;

import com.grantgupton.ml.perceptron.MultiLayerPerceptron;
import com.grantgupton.ml.perceptron.Perceptron;
import com.grantgupton.ml.perceptron.PerceptronData;
import com.grantgupton.ml.perceptron.SingleLayerPerceptron;

public class PerceptronVisualizer extends JFrame {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int[] layers;
	private int maxHeight;
	private Perceptron perceptron;

	private static final int CIRCLE_SIZE = 20;
	private static final int PADDING_X = 140;
	private static final int PADDING_Y = 20;

	public PerceptronVisualizer(Perceptron p) {
		this.perceptron = p;

		layers = perceptron.size();

		maxHeight = max(layers);

		setTitle("Perceptron Visualizer");
		setSize(layers.length * (CIRCLE_SIZE + PADDING_X), 60 + maxHeight * (CIRCLE_SIZE + PADDING_Y));
		setVisible(true);
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		this.addComponentListener(new ComponentListener() {

			@Override
			public void componentHidden(ComponentEvent arg0) {
			}

			@Override
			public void componentMoved(ComponentEvent arg0) {
			}

			@Override
			public void componentResized(ComponentEvent arg0) {

//				System.out.println("RESIZING EVENT:");
//				System.out.println("Frame Size: " + getWidth() + ", " + getHeight());
//				System.out.println("Pane1 Size: " + pane1.getWidth() + ", " + pane1.getHeight() + "\n");

//				repaint();
			}

			@Override
			public void componentShown(ComponentEvent arg0) {
			}
		});

	}

	private int max(int[] arr) {
		int m = arr[0];
		for (int i : arr) {
			if (i > m)
				m = i;
		}
		return m;
	}

	@Override
	public void paint(Graphics g) {
		Graphics2D g2d = (Graphics2D) g;

		for (int i = 0; i < layers.length; i++) {
			for (int j = 0; j < layers[i]; j++) {
				int x = 20 + (CIRCLE_SIZE + PADDING_X) * i;
				int y = 50 + (CIRCLE_SIZE + PADDING_Y) * j
						+ ((maxHeight - layers[i]) * (CIRCLE_SIZE / 2 + PADDING_Y / 2));

				g2d.drawOval(x, y, CIRCLE_SIZE, CIRCLE_SIZE);
				int textSpacing = 10;

				Rectangle r = new Rectangle(x, y, CIRCLE_SIZE, CIRCLE_SIZE);
//				drawCenteredString(g2d, i == 0 ? "Input" : "Bias: " + String.format("%.2f", perceptron.getBias(i, j)),
//						r, new Font(Font.SANS_SERIF, 1, 12), 0, -textSpacing);
//				drawCenteredString(g2d, "Val: " + String.format("%.2f", perceptron.getValue(i, j)), r,
//						new Font(Font.SANS_SERIF, 1, 12), 0, textSpacing);

				// If not output layer draw connecting lines
				if (i < layers.length - 1) {
					// For each neuron of current layer
					for (int m = 0; m < layers[i]; m++) {
						// Draw to each neuron of next layer
						for (int k = 0; k < layers[i + 1]; k++) {
							int y2 = 50 + (CIRCLE_SIZE + PADDING_Y) * k
									+ ((maxHeight - layers[i + 1]) * (CIRCLE_SIZE / 2 + PADDING_Y / 2))
									+ (CIRCLE_SIZE / 2);
							g2d.drawLine(x + CIRCLE_SIZE, y + (CIRCLE_SIZE / 2), x + (CIRCLE_SIZE + PADDING_X), y2);
							Rectangle r2 = new Rectangle(x + CIRCLE_SIZE, y2 - (textSpacing * m * 2), PADDING_X,
									PADDING_Y + (textSpacing * m));
//							drawCenteredString(g2d, String.format("%.2f", perceptron.getWeight(i, m, k)), r2,
//									new Font(Font.SANS_SERIF, 1, 12));
						}
					}
				}
			}
		}

	}

	/**
	 * Draw a String centered in the middle of a Rectangle.
	 *
	 * @param g    The Graphics instance.
	 * @param text The String to draw.
	 * @param rect The Rectangle to center the text in.
	 */
	public void drawCenteredString(Graphics2D g, String text, Rectangle rect, Font font, int dx, int dy) {
		// Get the FontMetrics
		FontMetrics metrics = g.getFontMetrics(font);
		// Determine the X coordinate for the text
		int x = rect.x + (rect.width - metrics.stringWidth(text)) / 2;
		// Determine the Y coordinate for the text (note we add the ascent, as in java
		// 2d 0 is top of the screen)
		int y = rect.y + ((rect.height - metrics.getHeight()) / 2) + metrics.getAscent();
		// Set the font
		g.setFont(font);
		// Draw the String
		g.drawString(text, x + dx, y + dy);
	}

	public void drawCenteredString(Graphics2D g, String text, Rectangle rect, Font font) {
		drawCenteredString(g, text, rect, font, 0, 0);
	}

	public static void main(String[] args) {
		SingleLayerPerceptron slp = new SingleLayerPerceptron(4);
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(new int[] {25, 20, 15, 10, 15, 20, 10});
//		PerceptronData[] trainData = { new PerceptronData(new double[] { 1, 0 }, new double[] { 1 }),
//				new PerceptronData(new double[] { 1, 1 }, new double[] { 0 }),
//				new PerceptronData(new double[] { 0, 1 }, new double[] { 1 }),
//				new PerceptronData(new double[] { 0, 0 }, new double[] { 0 }), };
		PerceptronVisualizer p = new PerceptronVisualizer(mlp);
//		p.repaint();
//		Consumer<String> con = (str) -> repaintHelper(p);

//		mlp.train(trainData, con);
	}

	private void setUp() {
//		offScreen = gc.createCompatibleImage(getSize().width, getSize().height, Transparency.TRANSLUCENT);
	}

	private static void repaintHelper(PerceptronVisualizer p) {
		p.repaint();
	}
}