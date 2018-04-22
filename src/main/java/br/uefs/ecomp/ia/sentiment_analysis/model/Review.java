package br.uefs.ecomp.ia.sentiment_analysis.model;

public class Review {

	private int stars;
	private String tile;
	private String comment;
	private int[] vector;

	public Review(int stars, String tile, String comment) {
		this.stars = stars;
		this.tile = tile;
		this.comment = comment;
	}

	public int getEstrelas() {
		return stars;
	}

	public void setStars(int stars) {
		this.stars = stars;
	}

	public String getTitle() {
		return tile;
	}

	public void setTitle(String title) {
		this.tile = title;
	}

	public String getComment() {
		return comment;
	}

	public void setComment(String comment) {
		this.comment = comment;
	}

	public int[] getVector() {
		return vector;
	}

	public void setVector(int[] vector) {
		this.vector = vector;
	}

	public boolean isNegative() {
		return stars < 3;
	}

	public boolean isPositive() {
		return stars > 3;
	}
}
