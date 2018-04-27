package br.uefs.ecomp.ia.sentiment_analysis;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.List;
import br.uefs.ecomp.ia.sentiment_analysis.model.Review;
import br.uefs.ecomp.ia.sentiment_analysis.util.BagOfWords;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.Perceptron;
import org.neuroph.util.TransferFunctionType;

public class App {

	public static String STOP_WORDS_FILE = "files/stopwords.txt";
	public static String INPUT_TRAINNING_FILE = "files/input_trainning.csv";
	public static String INPUT_VALIDATION_FILE = "files/input_validation.csv";
	public static String INPUT_TEST_FILE = "files/input_test.csv";
	public static String COMMENTS_FILE = "files/comments.csv";

	public static void main(String[] args) throws IOException {
		List<String> stopWords = loadStopWords();
		List<Review> reviews = loadReviews();
		BagOfWords bow = createBOW(stopWords, reviews);
		createVecReviews(reviews, bow);
		NeuralNetwork neuralNetwork = createSimpleMultilayerPerceptronNN(bow, (bow.getVocabullarySize())/2);
		trainingNeuralNetwork(neuralNetwork, reviews); //TODO substituir por trainingReviews
	}

	/**
	 * Cria uma rede neural multicamada.
	 * @param bow objeto bag of words
	 * @param hiddenNeurons numero de neuronoios na camada oculta
	 * @return rede neural
	 */
	public static NeuralNetwork createSimpleMultilayerPerceptronNN(BagOfWords bow, int hiddenNeurons){
		NeuralNetwork neuralNetwork = new MultiLayerPerceptron(TransferFunctionType.SIGMOID,
				bow.getVocabullarySize(), hiddenNeurons, 1);
		System.out.println("creating multilayer perceptron...");
		return neuralNetwork;
	}

	/***
	 * Método responsável por criar a a rede neural perceptron, entradas do tamanho do vocabulario
	 * e saída de tamanho 1. Função de transferência: sigmoide.
	 * @param bow bow contendo vocabulario
	 * @return rede neural do tipo perceptron simples.
	 */
	private static NeuralNetwork createSimplePerceptronNN(BagOfWords bow) {
		NeuralNetwork neuralNetwork = new Perceptron(bow.getVocabullarySize(),1, TransferFunctionType.SIGMOID);
		return neuralNetwork;
	}

	/**
	 * Treina rede neural com treino padrão,
	 * na perceptron: backpropagation
	 * @param neuralNetwork
	 * @param trainReviews
	 */
	private static void trainingNeuralNetwork(NeuralNetwork neuralNetwork, List<Review> trainReviews){
		//criando dataset de treinamento, entrada do tamanho do vacabulario e saída 1
		DataSet traingSet = new DataSet(neuralNetwork.getInputsCount(), 1);
		for (Review r: trainReviews){
			double[] output;
			double[] input = r.getVector();

			//para facilitar o trabalho da conversão da sigmoide
			if(r.isNegative())
				output=new double[]{0.01};
			else
				output=new double[]{0.99};

			//adiciona comentario na base de treinamento
			traingSet.add(new DataSetRow(input, output));
		}

		//falta descobrir que tipo de treinamento é esse kkkk, queira a Deus que seja o
		//supervisionado, amém??

		neuralNetwork.learn(traingSet);
	}
	private static List<String> loadStopWords() throws IOException {
		List<String> stopWords = new LinkedList<>();
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(STOP_WORDS_FILE), "UTF-8"))) {
			reader.lines().forEach((l) -> stopWords.add(l));
		}
		return stopWords;
	}

	private static List<Review> loadReviews() throws IOException {
		List<Review> reviews = new LinkedList<>();
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(INPUT_TRAINNING_FILE), "UTF-8"))) {
			reader.lines().forEach((l) -> {
				String[] line = l.split(";");
				reviews.add(new Review(Integer.parseInt(line[0]), line[1], line[2]));
			});
		}
		return reviews;
	}

	private static BagOfWords createBOW(List<String> stopWords, List<Review> reviews) {
		BagOfWords bow = new BagOfWords();
		bow.setStopWords(stopWords);
		bow.setType(BagOfWords.BINARY);
		reviews.forEach((r) -> bow.addLine(r.getComment()));
		bow.initialize();
		return bow;
	}

	private static void createVecReviews(List<Review> reviews, BagOfWords bow) {
		double[] vec;
		for (Review r : reviews) {
			vec = bow.createVec(r.getComment());
			r.setVector(vec);
		}
	}
}
