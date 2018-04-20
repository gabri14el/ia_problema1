package model;

public class Comentario {

    int estrelas; //qtd de estrelas
    boolean opiniao; //true positivo, false ruim
    String texto;
    String titulo;
    int [] vetorRepresentativo;

    public Comentario(int estrelas, boolean opiniao, String texto, String titulo) {
        this.estrelas = estrelas;
        this.opiniao = opiniao;
        this.texto = texto;
        this.titulo = titulo;
    }

    public void associarVetorRepresentativo(int[] vetor){
        vetorRepresentativo = vetor;
    }


    public String getTitulo() {
        return titulo;
    }

    public int getEstrelas() {
        return estrelas;
    }

    public boolean isOpiniao() {
        return opiniao;
    }

    public String getTexto() {
        return texto;
    }
}
