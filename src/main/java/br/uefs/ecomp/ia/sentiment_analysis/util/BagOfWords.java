package br.uefs.ecomp.ia.sentiment_analysis.util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import br.uefs.ecomp.ia.sentiment_analysis.App;
import br.uefs.ecomp.ia.sentiment_analysis.model.Review;

/**
 * Classe utilizada para realizar a transformação de frases para um vetor numérico
 * 
 * @author Matob
 *
 */
public class BagOfWords {

	public static int BINARY = 1;
	public static int TERM_FREQUENCY = 2;

	private static double FREQUENCY_TO_IGNORE_WORDS = 1;
	private static boolean FREQUENCY_BY_DOC = true;

	private List<String> stopWords; // Lista contendo as palavras que serão desconsideradas (passa pelo método clean)
	private List<String> text; // Lista contendo todas as frases de entrada
	private List<String> vocabullary; // Lista que será usada para armazenar as palavras distintas de text
	private int type; // Define a abordagem que será utilizara para a criação de vetores (BINARY, TF)

	private boolean debug = false;

	public BagOfWords() {
		vocabullary = new ArrayList<>();
		text = new LinkedList<>();
		stopWords = new LinkedList<>();
	}

	public void setStopWords(List<String> stopWords) {
		this.stopWords.clear();
		stopWords.forEach((s) -> this.stopWords.add(clean(s))); // Limpa as stop words
	}

	public void setType(int type) {
		this.type = type;
	}

	public void addLine(String line) {
		text.add(line);
	}

	/**
	 * Inicializa o vocabulário a partir da lista text.Deve ser chamado após ter sido inserida todas as frases
	 * Percorre todas as strings contidas em text, "limpando-as" e efetuando uma separação por espaços em branco
	 * e ignorando as palavras presentes em stopWords. Cada palavra é armazenada em um mapa individualmente junto
	 * com sua contagem de ocorrências
	 * 
	 */
	public void initialize() {
		Map<String, Integer> map = new HashMap<>();
		vocabullary.clear();

		List<String> counted = new LinkedList<>();
		List<String> words;
		for (String t : text) {
			counted.clear();
			t = clean(t);

			words = new LinkedList<>(Arrays.asList(t.split("\\s")));
			words.removeAll(stopWords);
			words.remove("");
			words.remove(" ");

			if (FREQUENCY_BY_DOC) {
				for (String w : words) {
					if (!counted.contains(w)) {
						map.put(w, (map.containsKey(w)) ? (map.get(w) + 1) : 1);
						counted.add(w);
					}
				}
			} else {
				for (String w : words)
					map.put(w, (map.containsKey(w)) ? (map.get(w) + 1) : 1);
			}
		}

		// Remove as palavras com baixa frequência
		Set<String> wordSet = map.keySet();
		List<String> removed = new LinkedList<>(wordSet);
		wordSet.removeIf((w) -> map.get(w) <= FREQUENCY_TO_IGNORE_WORDS);
		vocabullary.addAll(wordSet);

		removed.removeAll(wordSet);
		if (debug) {
			System.out.println("Removidos: " + Arrays.toString(removed.toArray()));
			System.out.println("Quantidade: " + removed.size());
			System.out.println("Vocabulário: " + Arrays.toString(wordSet.toArray()));
			System.out.println("Quantidade: " + wordSet.size());
		}
	}

	/**
	 * Cria um vetor representativo a partir de uma frase de entrada com base no
	 * vocabulario criado anteriormente. O vetor pode usar a abordagem binária
	 * ou de contagem de ocorrências.
	 */
	public double[] createVec(String line) {
		String l = clean(line);
		List<String> words = new LinkedList<>(Arrays.asList(l.split("\\s")));
		words.removeAll(stopWords);

		if (type == BINARY)
			return createBinaryVec(words);
		else if (type == TERM_FREQUENCY)
			return createTFVec(words);

		return new double[vocabullary.size()];
	}

	private double[] createBinaryVec(List<String> words) {
		double[] vec = new double[vocabullary.size()];
		for (String w : words) {
			if (vocabullary.contains(w))
				vec[vocabullary.indexOf(w)] = 1;
		}
		return vec;
	}

	private double[] createTFVec(List<String> words) {
		double[] vec = new double[vocabullary.size()];
		for (String w : words) {
			if (vocabullary.contains(w))
				vec[vocabullary.indexOf(w)]++;
		}
		return vec;
	}

	/**
	 * Efetua algumas manipulações de strings para "limpá-la". O propósito
	 * é reduzir a quantidade de palavras repetidas escritas de forma errada
	 * e também de ignorar possíveis palavras inúteis.
	 */
	private String clean(String t) {
		t = t.toLowerCase();
		t = Normalizer.normalize(t, Normalizer.Form.NFD);
		t = t.replaceAll("[^(\\w|\\s)]", ""); // Remove qualquer coisa fora da ascii
		t = t.replaceAll("[^(\\w|\\s)]", ""); // Remove qualquer coisa fora da ascii
		t = t.replaceAll("[\\p{InCombiningDiacriticalMarks}]", ""); // Remove acentuação
		t = t.replaceAll("\\d", ""); // Remove números
		t = t.replaceAll("(([A-Za-z])(\\2)+)", "$2"); // Remove caracteres duplicados, como aa, ee, ii...
		t = t.replace('(', ' '); // Remove parenteses
		t = t.replace(')', ' '); // Remove parenteses
		t = t.replaceAll("\\s.{1}\\s", " "); // Remove "palavras" com apenas um caractere.
		t = t.replaceAll("\\s+", " "); // Remove múltiplos espaços em branco.
		t = t.trim();

		return t;
	}

	/**
	 *
	 * Método que retorna tamanho do vocabulário
	 * 
	 * @return
	 */
	public int getVocabullarySize() {
		return vocabullary.size();
	}

	public void setDebug(boolean debug) {
		this.debug = debug;
	}

	public static void main(String[] args) throws IOException {
		List<Review> reviews = new LinkedList<>();
		String[] line;
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(App.INPUT_TRAINNING_FILE), "UTF-8"))) {
			while (reader.ready()) {
				line = reader.readLine().split(";");
				reviews.add(new Review(Integer.parseInt(line[0]), line[1], line[2]));
			}
		}

		List<String> stopWords = new LinkedList<>();
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(App.STOP_WORDS_FILE), "UTF-8"))) {
			reader.lines().forEach((l) -> stopWords.add(l));
		}

		BagOfWords bow = new BagOfWords();
		bow.setDebug(true);
		bow.setStopWords(stopWords);
		bow.setType(BagOfWords.BINARY);
		reviews.forEach((r) -> bow.addLine(r.getComment()));
		bow.initialize();
	}
}
