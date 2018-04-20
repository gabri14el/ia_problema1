package model;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.Normalizer;
import java.util.*;
import java.util.regex.Pattern;

public class BagOfWords {

    ArrayList<String> listaDePalavras;
    LinkedList<Comentario> comentarios_treinamento;
    String arquivoComentarios;
    String arquivoStopWords;
    LinkedList<String> stopWords;

    List<Comentario> positivos;
    List<Comentario> negativos;

    public static int VETOR_BINARIO = 0;
    public static int VETOR_TF = 1;
    public static int VETOR_ITF = 2;


    public BagOfWords(String arquivoComentarios, String arquivoStopWords){
        listaDePalavras = new ArrayList<String>();
        comentarios_treinamento = new LinkedList<Comentario>();
        this.arquivoComentarios = arquivoComentarios;
        this.arquivoStopWords = arquivoStopWords;
        stopWords = new LinkedList<String>();

        positivos = new ArrayList<>();
        negativos = new ArrayList<>();


        carregaStopWords();
        leComentarios();
        criaVocabulario();
        gerarVetorRepresentativoDosComentarios(VETOR_BINARIO);

        System.out.println("quantidade de comentarios positivos (treinamento): "+positivos.size());
        System.out.println("quantidade de comentarios negativos (treinamento): "+negativos.size());
    }

    //gerar vocabulario:

        //ler do arquivo os comentários
        //separar as palavras numa lista encadeada
        //remover dessa lista das palavras as stopwords em pt-br
        //remover sinais, virgulas, pontos, exclamação, hifen e etc
        //gerar um array a partir dessa lista

    //gerar comentario

    private void leComentarios(){
        try {
            BufferedReader reader = new BufferedReader(new FileReader(arquivoComentarios));
            reader.readLine();

            while(reader.ready()){
                String temp = reader.readLine();
                StringTokenizer tokenizer = new StringTokenizer(temp, ";");
                int estrelas = Integer.parseInt(tokenizer.nextToken());
                boolean opinião;
                if(estrelas > 3)
                    opinião = true;
                else
                    opinião = false;
                String aux = tokenizer.nextToken();
                String titulo = limpaString(aux);
                aux = tokenizer.nextToken();
                String texto = limpaString(aux);


                comentarios_treinamento.add(new Comentario(estrelas, opinião, texto, titulo));
            }
        } catch (FileNotFoundException e) {
            System.err.println(e);
        } catch (IOException e) {
            System.err.println(e);
        }
    }

    private void criaVocabulario(){

        for (Comentario comentario: comentarios_treinamento){
            StringTokenizer tokenizer = new StringTokenizer(comentario.getTexto());
            while(tokenizer.hasMoreTokens()){
                String a = tokenizer.nextToken();
                if(!listaDePalavras.contains(a))
                    listaDePalavras.add(a);
            }
        }

        listaDePalavras.removeAll(stopWords);
        System.out.println("tamanho do vocabulario: "+listaDePalavras.size());
    }

    private void carregaStopWords(){
        try {
            BufferedReader reader = new BufferedReader(new FileReader(arquivoStopWords));
            while(reader.ready())
                stopWords.add(deAccent(reader.readLine()));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    //source: https://pt.stackoverflow.com/questions/42/como-remover-acentos-e-outros-sinais-gr%C3%A1ficos-de-uma-string-em-java
    private static String deAccent(String str) {
        String nfdNormalizedString = Normalizer.normalize(str, Normalizer.Form.NFD);
        Pattern pattern = Pattern.compile("\\p{InCombiningDiacriticalMarks}+");
        return pattern.matcher(nfdNormalizedString).replaceAll("");
    }

    //limpa a string
    private String limpaString(String str){
        return deAccent(str).replaceAll("[^a-zA-Z\\s]", "")
                .toLowerCase();
    }


    /**
     * Retorna um vetor representativo de uma determinada string
     * @param str
     * @param tipoDeVetor
     * @return
     */
    public int[] criaVetor(String str, int tipoDeVetor){
        int[] vetor = new int[listaDePalavras.size()];
        str = limpaString(str);
        List<String> palavras = Arrays.asList(str.split("\\s"));
        //palavras.removeAll(stopWords);
        if(tipoDeVetor == VETOR_BINARIO)
            return vetorBinario(vetor, palavras);
        else if(tipoDeVetor == VETOR_TF)
            return vetorTF(vetor, palavras);
        else
            return vetorITF(vetor, palavras);
    }

    private int[] vetorITF(int[] vetor, List<String> palavras) {
        //nao suportado
        return vetor;
    }

    /**
     * Retorna um vetor com a quantidade de palavras no documento
     * @param vetor
     * @param palavras
     * @return
     */
    private int[] vetorTF(int[] vetor, List<String> palavras) {
        for(String palavra: palavras){
            if(listaDePalavras.contains(palavra)){
                vetor[listaDePalavras.indexOf(palavra)]++;
            }
        }
        return vetor;
    }

    /**
     * Retorna um vetor binario
     * @param vetor
     * @param palavras
     * @return
     */
    private int[] vetorBinario(int[] vetor, List<String> palavras){
        for(String palavra: palavras){
            if(listaDePalavras.contains(palavra)){
                vetor[listaDePalavras.indexOf(palavra)]=1;
            }
        }
        return vetor;
    }

    //seta o vetor representativo de 3000 comentários para treino da
    //rede neural
    public void gerarVetorRepresentativoDosComentarios(int tipo){
        int positivo = 0;
        int negativo = 0;
        boolean gerar = true;
        for(Comentario comentario: comentarios_treinamento){
            if(comentario.isOpiniao() && positivo <= 3000){
                gerar = true;
                positivo++;
                positivos.add(comentario);
            }
            else if(comentario.isOpiniao() && positivo > 3000)
                gerar = false;
            else if(!comentario.isOpiniao() && negativo <= 3000){
                negativo++;
                gerar = true;
                negativos.add(comentario);
            }
            else{
                gerar = false;
            }

            if(gerar)
                comentario.associarVetorRepresentativo(criaVetor(comentario.getTexto(), tipo));

            if(positivo > 3000 && negativo > 3000)
                return;
        }
    }
}
