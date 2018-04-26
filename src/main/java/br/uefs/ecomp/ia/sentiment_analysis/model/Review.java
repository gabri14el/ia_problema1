package br.uefs.ecomp.ia.sentiment_analysis.model;

public class Review {

	private int stars;
	private String title;
	private String comment;
	private double[] vector;

	public Review(int stars, String title, String comment) {
		this.stars = stars;
		this.title = title;
		this.comment = comment;
	}

	public int getEstrelas() {
		return stars;
	}

	public void setStars(int stars) {
		this.stars = stars;
	}

	public String getTitle() {
		return title;
	}

	public void setTitle(String title) {
		this.title = title;
	}

	public String getComment() {
		return comment;
	}

	public void setComment(String comment) {
		this.comment = comment;
	}

	public double[] getVector() {
		return vector;
	}

	public void setVector(double[] vector) {
		this.vector = vector;
	}

	public boolean isNegative() {
		return stars < 3;
	}

	public boolean isPositive() {
		return stars > 3;
	}

	@Override
	public String toString() {
		return String.format("%d;%s;%s", stars, title, comment);
	}
}
