package br.uefs.ecomp.ia.sentiment_analysis;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import br.uefs.ecomp.ia.sentiment_analysis.model.Review;
import br.uefs.ecomp.ia.sentiment_analysis.util.BagOfWords;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.Perceptron;
import org.neuroph.nnet.learning.BackPropagation;
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
		System.out.println("criando NN multilayer perceptron...");
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


		System.out.println("iniciando treinamento da rede neural...");
		BackPropagation backPropagation = new BackPropagation();
		backPropagation.setMaxIterations(1000);
		//backPropagation.setMaxError(0.01);
		backPropagation.addListener(new LearningEventListener() {
			@Override
			public void handleLearningEvent(LearningEvent learningEvent) {
				System.out.println(learningEvent.getEventType().name());
				if(learningEvent.getEventType().equals(LearningEvent.Type.LEARNING_STOPPED))
					System.out.println("o erro foi: "+ backPropagation.getTotalNetworkError());
				//guardar junto os pesos dos neuronios quando alcancar o menor erro
			}
		});
		long inicio = System.currentTimeMillis();
		neuralNetwork.learn(traingSet, backPropagation);
		long fim = System.currentTimeMillis();

		System.out.println("treinamento terminado em :" + (fim-inicio)/1000 + "segundos");

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
		System.out.println("criando vetores para os comentarios...");
		double[] vec;
		for (Review r : reviews) {
			vec = bow.createVec(r.getComment());
			r.setVector(vec);
		}
	}


	/**
	 * Método temporário so pra testar as brincadeiras com a rede neural =)
	 * @throws IOException
	 */
	private static void brincandoComRN() throws IOException {
		List<String> stopWords = loadStopWords();
		List<Review> comentarios = new ArrayList<>();


		comentarios.add(new Review(5, "top"));
		comentarios.add(new Review(5, "produto legal"));
		comentarios.add(new Review(5, "produto bom"));
		comentarios.add(new Review(5, "gostei"));

		comentarios.add(new Review(1, "ruim"));
		comentarios.add(new Review(1, "nao gostei"));
		comentarios.add(new Review(1, "produto de qualidade duvidosa"));
		comentarios.add(new Review(1, "produto meia boca"));
		comentarios.add(new Review(1, "pessimo me arrependi muito"));

		BagOfWords bow = createBOW(stopWords, comentarios);
		createVecReviews(comentarios, bow);
		NeuralNetwork neuralNetwork = createSimpleMultilayerPerceptronNN(bow, bow.getVocabullarySize()/2);
		trainingNeuralNetwork(neuralNetwork, comentarios);

	}
	
}
