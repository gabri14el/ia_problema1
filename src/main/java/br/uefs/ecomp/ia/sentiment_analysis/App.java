package br.uefs.ecomp.ia.sentiment_analysis;

import java.io.*;
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
import org.neuroph.nnet.learning.DynamicBackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.nnet.learning.ResilientPropagation;
import org.neuroph.util.TransferFunctionType;
import br.uefs.ecomp.ia.sentiment_analysis.model.ErrorData;
import br.uefs.ecomp.ia.sentiment_analysis.model.Review;
import br.uefs.ecomp.ia.sentiment_analysis.util.BagOfWords;

import javax.swing.plaf.synth.SynthOptionPaneUI;

public class App {

	public static String STOP_WORDS_FILE = "files/stopwords.txt";
	public static String INPUT_TRAINNING_FILE = "files/input_trainning.csv";
	public static String INPUT_VALIDATION_FILE = "files/input_validation.csv";
	public static String INPUT_TEST_FILE = "files/input_test.csv";
	public static String COMMENTS_FILE = "files/comments.csv";

	private static final double NEGATIVE_WEIGHT = 0.01;
	private static final double POSITIVE_WEIGHT = 0.99;

	private static int DYNAMIC_MOMENTUM_BACKPROPAGATION = 1;
	private static int BACKPROPAGATION = 2;
	private static int RESILIENT_PROPAGATION = 3;

	public static void main(String[] args) throws IOException {
		List<Review> test = load(INPUT_TEST_FILE);
		List<Review> validation = load(INPUT_VALIDATION_FILE);
		List<Review> trainning = load(INPUT_TRAINNING_FILE);
		List<String> stopWords = loadStopWords();

		BagOfWords bow = createBOW(stopWords, trainning);
		createVecReviews(test, bow);
		createVecReviews(validation, bow);
		createVecReviews(trainning, bow);

		System.out.println("tamanho do vocabulario: "+bow.getVocabullarySize()); //deixa essa porra aí
		System.out.println("tamanho da validacao: "+validation.size());
		System.out.println("tamanho do teste: "+test.size());
		System.out.println("tamanho do treino: " +trainning.size());

		System.out.println("###################################  TREINAMENTO 1  ###################################");
		int hiddenLayerSize = 50;
		System.out.println("quantidade de neuronios da camada oculta: "+hiddenLayerSize);
		System.out.println("treinamento dynamic backpropagation");
		NeuralNetwork neuralNetwork = createSimpleMultilayerPerceptronNN(bow, hiddenLayerSize);
		initializeNeurons(neuralNetwork);
		Double[] pesos = neuralNetwork.getWeights();
		trainingNeuralNetwork(neuralNetwork, trainning, validation,0.3, 0.01,
				0.05, 200, 0.9, DYNAMIC_MOMENTUM_BACKPROPAGATION); //TODO substituir por trainingReviews
		double[][] resultado = testNeuralNetwork(test, neuralNetwork);
		printResults(resultado);




		System.out.println("\n\n");
		System.out.println("###################################  TREINAMENTO 7  ###################################");
		hiddenLayerSize = 50;
		System.out.println("treinamento dynamic backpropagation");
		System.out.println("quantidade de neuronios da camada oculta: "+hiddenLayerSize);
		neuralNetwork = createSimpleMultilayerPerceptronNN(bow, hiddenLayerSize);
		neuralNetwork.setWeights(DoubleVectorToPrimitive(pesos));
		trainingNeuralNetwork(neuralNetwork, trainning, validation,0.3, 0.01,
				0.05, 200, 0.9, BACKPROPAGATION); //TODO substituir por trainingReviews
		resultado = testNeuralNetwork(test, neuralNetwork);
		printResults(resultado);

	}

	/**
	 * Método para imprimir resultados
	 * @param results
	 */
	private static void printResults(double[][] results){
		System.out.println("==================RESULTADOS==================");
		System.out.println("esperado\tobtido");
		int acertos = 0;
		for(int i=0; i<results[0].length; i++){
			//System.out.println(results[0][i]+"\t"+results[1][i]);
			if((results[0][i]> 0.5 && results[1][i]>0.5) || (results[0][i]<= 0.5 && results[1][i]<=0.5))
				acertos++;
		}
		System.out.println("----------------------");
		System.out.println("total de comentarios: "+results[0].length);
		System.out.println("total de acertos: "+acertos);
		//System.out.println("porcentagem de acerto: "+(acertos/results[0].length)*100);
		System.out.println("----------------------");
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
	private static void trainingNeuralNetwork(NeuralNetwork neuralNetwork, List<Review> trainReviews, List<Review> validationReviews,
											  double maxLearningRate, double maxError, double maxLearningRateChange, int maxEpoch, double maxMomentum, int treinamento) {
		Double[][] bestWeights = new Double[][] { {} };
		double[] minValidationError = new double[] { -1 };
		List<ErrorData> errors = new LinkedList<>();
		//criando dataset de treinamento, entrada do tamanho do vacabulario e saída 1
		DataSet traingSet = List2DataSet(trainReviews, neuralNetwork.getInputsCount(), neuralNetwork.getOutputsCount());
		DataSet validationSet = List2DataSet(validationReviews, neuralNetwork.getInputsCount(),
				neuralNetwork.getOutputsCount());

		System.out.println("sorteando pesos iniciais dos neuronios...");
		//initializeNeurons(neuralNetwork);

		System.out.println("iniciando treinamento da rede neural...");
		System.out.println("taxa de aprendizado maxima: "+maxLearningRate);
		System.out.println("maxima mudança na taxa de aprendizado: "+maxLearningRateChange);
		System.out.println("momentum maximo: "+maxMomentum);
		System.out.println("erro maximo: "+maxError);
		System.out.println();


		BackPropagation backPropagation;
		if(treinamento == 1){
			backPropagation = new DynamicBackPropagation();
			backPropagation.setMaxIterations(maxEpoch); //quantidade maxima de epocas
			backPropagation.setMaxError(maxError); //erro maximo permitido para parar o treinamento
			//backPropagation.setLearningRate(learningRate);//taxa de aprendizado
			//backPropagation.setBatchMode(true);

			((DynamicBackPropagation)backPropagation).setMaxLearningRate(maxLearningRate);
			((DynamicBackPropagation)backPropagation).setLearningRateChange(maxLearningRateChange);
			((DynamicBackPropagation)backPropagation).setMaxMomentum(maxMomentum);
		}
		else if(treinamento == 2){
			backPropagation = new BackPropagation();
			backPropagation.setMaxIterations(maxEpoch); //quantidade maxima de epocas
			backPropagation.setMaxError(maxError); //erro maximo permitido para parar o treinamento
			backPropagation.setLearningRate(maxLearningRate);//taxa de aprendizado
			//backPropagation.setBatchMode(true);
		}
		else{
			backPropagation = new ResilientPropagation();
			((ResilientPropagation)backPropagation).setMaxIterations(maxEpoch); //quantidade maxima de epocas
			//((ResilientPropagation)backPropagation).setMaxError(maxError); //erro maximo permitido para parar o treinamento
			//((ResilientPropagation)backPropagation).setLearningRate(maxLearningRate);//taxa de aprendizado
			((ResilientPropagation)backPropagation).setInitialDelta(0.1);
			((ResilientPropagation)backPropagation).setMaxDelta(50);
			((ResilientPropagation)backPropagation).setMinDelta(0.000001);
			((ResilientPropagation)backPropagation).setIncreaseFactor(0.5);
			((ResilientPropagation)backPropagation).setDecreaseFactor(1.2);


		}


		System.out.println("Erro Mínimo\t\tErro Médio\t\tErro Treino");
		long time = System.currentTimeMillis();
		backPropagation.addListener(new LearningEventListener() {

			@Override
			public void handleLearningEvent(LearningEvent learningEvent) {
				if (learningEvent.getEventType().equals(LearningEvent.Type.LEARNING_STOPPED)){
					System.out.println("==================FIM DE TREINAMENTO==================");
					System.out.println("menor erro de validacao: "+minValidationError[0]);
					System.out.println("quantidade de epocas: "+backPropagation.getCurrentIteration());
					System.out.println("erro total na rede neural: "+backPropagation.getTotalNetworkError());
				}


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

					//System.out.println("Time (s): " + ((System.currentTimeMillis() - time) / 1000));
					//System.out.format("%11d %11d %11d", minValidationError, mediumValidationError, trainingError); ta errado
					System.out.println(mediumValidationError+"\t\t"+trainingError);
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

		System.out.println("\n\niniciando os testes...");
		double[][] resultados = new double[2][testSet.getRows().size()];
		for (int i = 0; i < testSet.getRows().size(); i++) {
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
