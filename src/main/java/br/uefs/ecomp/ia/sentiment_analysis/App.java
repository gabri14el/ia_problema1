package br.uefs.ecomp.ia.sentiment_analysis;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.List;
import br.uefs.ecomp.ia.sentiment_analysis.model.Review;
import br.uefs.ecomp.ia.sentiment_analysis.util.BagOfWords;

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
		int[] vec;
		for (Review r : reviews) {
			vec = bow.createVec(r.getComment());
			r.setVector(vec);
		}
	}
}
