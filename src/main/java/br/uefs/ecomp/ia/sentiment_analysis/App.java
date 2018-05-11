package br.uefs.ecomp.ia.sentiment_analysis;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.Perceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import br.uefs.ecomp.ia.sentiment_analysis.model.ErrorData;
import br.uefs.ecomp.ia.sentiment_analysis.model.Review;
import br.uefs.ecomp.ia.sentiment_analysis.util.BagOfWords;

public class App {

	public static String STOP_WORDS_FILE = "files/stopwords.txt";
	public static String INPUT_TRAINNING_FILE = "files/input_trainning.csv";
	public static String INPUT_VALIDATION_FILE = "files/input_validation.csv";
	public static String INPUT_TEST_FILE = "files/input_test.csv";
	public static String COMMENTS_FILE = "files/comments.csv";

	private static final double NEGATIVE_WEIGHT = 0.01;
	private static final double POSITIVE_WEIGHT = 0.99;

	public static void main(String[] args) throws IOException {
		List<Review> test = load(INPUT_TEST_FILE);
		List<Review> validation = load(INPUT_VALIDATION_FILE);
		List<Review> trainning = load(INPUT_TRAINNING_FILE);
		List<String> stopWords = loadStopWords();

		BagOfWords bow = createBOW(stopWords, trainning);
		createVecReviews(test, bow);
		createVecReviews(validation, bow);
		createVecReviews(trainning, bow);

		NeuralNetwork neuralNetwork = createSimpleMultilayerPerceptronNN(bow, (bow.getVocabullarySize()));
		trainingNeuralNetwork(neuralNetwork, trainning, validation,0.3, 0.003, 50); //TODO substituir por trainingReviews
		double[][] resultado = testNeuralNetwork(test, neuralNetwork);
	}

	private static List<Review> load(String fileName) throws IOException {
		List<Review> reviews = new LinkedList<>();
		String[] line;
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"))) {
			while (reader.ready()) {
				line = reader.readLine().split(";");
				reviews.add(new Review(Integer.parseInt(line[0]), line[1], line[2]));
			}
		}
		return reviews;
	}

	/**
	 * Cria uma rede neural multicamada.
	 * 
	 * @param bow
	 *            objeto bag of words
	 * @param hiddenNeurons
	 *            numero de neuronoios na camada oculta
	 * @return rede neural
	 */
	public static NeuralNetwork createSimpleMultilayerPerceptronNN(BagOfWords bow, int hiddenNeurons) {
		NeuralNetwork neuralNetwork = new MultiLayerPerceptron(TransferFunctionType.SIGMOID,
				bow.getVocabullarySize(), hiddenNeurons, 1);
		System.out.println("criando NN multilayer perceptron...");
		return neuralNetwork;
	}

	/***
	 * Método responsável por criar a a rede neural perceptron, entradas do tamanho do vocabulario
	 * e saída de tamanho 1. Função de transferência: sigmoide.
	 * 
	 * @param bow
	 *            bow contendo vocabulario
	 * @return rede neural do tipo perceptron simples.
	 */
	private static NeuralNetwork createSimplePerceptronNN(BagOfWords bow) {
		NeuralNetwork neuralNetwork = new Perceptron(bow.getVocabullarySize(), 1, TransferFunctionType.SIGMOID);
		return neuralNetwork;
	}

	private static double[] DoubleVectorToPrimitive(Double[] vector) {
		double primitiveVector[] = new double[vector.length];

		for (int i = 0; i < vector.length; i++) {
			primitiveVector[i] = vector[i].doubleValue();
		}
		return primitiveVector;
	}

	/**
	 * Treina rede neural com treino padrão,
	 * na perceptron: backpropagation
	 * 
	 * @param neuralNetwork
	 * @param trainReviews
	 */
	private static void trainingNeuralNetwork(NeuralNetwork neuralNetwork, List<Review> trainReviews, List<Review> validationReviews, double learningRate, double maxError, int maxEpoch) {
		Double[][] bestWeights = new Double[][] { {} };
		double[] minValidationError = new double[] { -1 };
		List<ErrorData> errors = new LinkedList<>();
		//criando dataset de treinamento, entrada do tamanho do vacabulario e saída 1
		DataSet traingSet = List2DataSet(trainReviews, neuralNetwork.getInputsCount(), neuralNetwork.getOutputsCount());

		DataSet validationSet = List2DataSet(validationReviews, neuralNetwork.getInputsCount(),
				neuralNetwork.getOutputsCount());

		System.out.println("sorteando pesos iniciais dos neuronios...");
		initializeNeurons(neuralNetwork);

		System.out.println("iniciando treinamento da rede neural...");

		BackPropagation backPropagation = new BackPropagation();
		backPropagation.setMaxIterations(maxEpoch); //quantidade maxima de epocas
		backPropagation.setMaxError(maxError); //erro maximo permitido para parar o treinamento
		backPropagation.setLearningRate(learningRate);//taxa de aprendizado 

		System.out.println("Erro Mínimo Erro Médio  Erro Treino");
		long time = System.currentTimeMillis();
		backPropagation.addListener(new LearningEventListener() {

			@Override
			public void handleLearningEvent(LearningEvent learningEvent) {
				System.out.println(learningEvent.getEventType().name());
				if (learningEvent.getEventType().equals(LearningEvent.Type.LEARNING_STOPPED))
					System.out.println("o erro foi: " + backPropagation.getTotalNetworkError());
				//guardar junto os pesos dos neuronios quando alcancar o menor erro

				else if (learningEvent.getEventType().equals(LearningEvent.Type.EPOCH_ENDED)) {
					double validationError = 0;
					double mediumValidationError;

					//passa por todas as linhas de trainamento, salvando o erro quadrático
					for (DataSetRow r : validationSet) {
						neuralNetwork.setInput(r.getInput());
						neuralNetwork.calculate();
						double[] output = neuralNetwork.getOutput();
						validationError += (r.getDesiredOutput()[0] - output[0]) * (r.getDesiredOutput()[0] - output[0]); //erro quadratico, por isso elevar ao quadrado       
					}

					//calcula o erro quadrático médio
					mediumValidationError = validationError / validationSet.size(); //erro de validacao
					//pega o erro de treinamento 
					double trainingError = backPropagation.getTotalNetworkError(); //erro de treino

					ErrorData errorData = new ErrorData(mediumValidationError,
							trainingError, backPropagation.getCurrentIteration()); //objeto que usaremos para construir os gráficos
					errors.add(errorData);

					if (minValidationError[0] == -1) {
						minValidationError[0] = mediumValidationError;
						bestWeights[0] = neuralNetwork.getWeights();
					} else if (mediumValidationError <= minValidationError[0]) {
						minValidationError[0] = mediumValidationError;
						bestWeights[0] = neuralNetwork.getWeights();
					}

					System.out.println("Time (s): " + ((System.currentTimeMillis() - time) / 1000));
					System.out.format("%11d %11d %11d", minValidationError, mediumValidationError, trainingError);
				}
			}
		});

		neuralNetwork.learn(traingSet, backPropagation);
		neuralNetwork.setWeights(DoubleVectorToPrimitive(bestWeights[0]));
	}

	private static List<String> loadStopWords() throws IOException {
		List<String> stopWords = new LinkedList<>();
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(STOP_WORDS_FILE), "UTF-8"))) {
			reader.lines().forEach((l) -> stopWords.add(l));
		}
		return stopWords;
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
	 * Métodos responsável por colocar pesos aleatórios nos neurônios.
	 * 
	 * @param neuralNetwork
	 */
	private static void initializeNeurons(NeuralNetwork neuralNetwork) {
		Layer hiddenLayer = neuralNetwork.getLayerAt(1);
		List<Neuron> neurons = hiddenLayer.getNeurons();

		Random random = new Random();
		for (Neuron n : neurons) {
			n.initializeWeights(random.nextGaussian());
			//System.out.println(n.getWeights());
		}
	}

	/**
	 * Converte uma lista de de reviews num objeto DataSet
	 * 
	 * @param reviews
	 *            lista de reviews
	 * @param inputCount
	 *            dimensão da entrada
	 * @param outputCount
	 *            dimensão da saída
	 * @return objeto dataset
	 */
	private static DataSet List2DataSet(List<Review> reviews, int inputCount, int outputCount) {
		DataSet set = new DataSet(inputCount, outputCount);
		for (Review r : reviews) {
			double[] output;
			double[] input = r.getVector();

			//para facilitar o trabalho da conversão da sigmoide
			if (r.isNegative())
				output = new double[] { NEGATIVE_WEIGHT };
			else
				output = new double[] { POSITIVE_WEIGHT };

			//adiciona comentario na base de treinamento
			set.add(new DataSetRow(input, output));
		}
		return set;
	}

	private static double[][] testNeuralNetwork(List<Review> test, NeuralNetwork neuralNetwork){
		DataSet testSet = List2DataSet(test, neuralNetwork.getInputsCount(), neuralNetwork.getOutputsCount());
		double[][] resultados = new double[2][testSet.size()];
		for (int i = 0; i < testSet.size(); i++) {
			neuralNetwork.setInput(testSet.get(i).getInput());
			neuralNetwork.calculate();
			double desejado = testSet.get(i).getDesiredOutput()[0];
			double saida = neuralNetwork.getOutput()[0];
			resultados[0][i] = desejado;
			resultados[1][i] = saida;
		}

		return resultados;
	}
	

}
