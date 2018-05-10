package br.uefs.ecomp.ia.sentiment_analysis.util;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class BagOfWords {

	public static int BINARY = 1;
	public static int TERM_FREQUENCY = 2;

	private static double FREQUENCY_TO_IGNORE_WORDS = 5;

	private List<String> stopWords;
	private List<String> text;
	private List<String> vocabullary;
	private int type;

	public BagOfWords() {
		vocabullary = new ArrayList<>();
		text = new LinkedList<>();
		stopWords = new LinkedList<>();
	}

	public void setStopWords(List<String> stopWords) {
		this.stopWords.clear();
		stopWords.forEach((s) -> this.stopWords.add(clean(s))); // Limpa os stop words
	}

	public void setType(int type) {
		this.type = type;
	}

	public void addLine(String line) {
		text.add(line);
	}

	public void initialize() {
		Map<String, Integer> map = new HashMap<>();
		vocabullary.clear();

		for (String t : text) {
			t = clean(t);
			List<String> words = new LinkedList<>(Arrays.asList(t.split("\\s")));
			words.removeAll(stopWords);

			for (String w : words)
				map.put(w, (map.containsKey(w)) ? (map.get(w) + 1) : 1);
		}

		// Remove as palavras com baixa frequência
		Set<String> words = map.keySet();
		words.removeIf((w) -> map.get(w) < FREQUENCY_TO_IGNORE_WORDS);
		vocabullary.addAll(words);

		System.out.println("Tamanho do vocabulário: " + vocabullary.size());
	}

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

	private String clean(String t) {
		t = Normalizer.normalize(t, Normalizer.Form.NFD);
		t = t.replaceAll("[^\\p{ASCII}]", ""); // Remove qualquer coisa fora da ascii
		t = t.replaceAll("[\\p{InCombiningDiacriticalMarks}]", ""); // Remove acentuação
		t = t.replaceAll("\\d", ""); // Remove números
		t = t.replaceAll("(([A-Za-z])(\\2)+)", "$2"); // Remove caracteres duplicados, como aa, ee, ii...
		t = t.replaceAll("\\s+", " ");
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
}
