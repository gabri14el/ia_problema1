/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package br.uefs.ecomp.ia.sentiment_analysis.model;

/**
 *
 * @author gabriel
 */
public class ErrorData {
    
    double errorMax;
    double errorTraining;
    int epoch;

    public ErrorData(double errorMax, double errorTraining, int epoch) {
        this.errorMax = errorMax;
        this.errorTraining = errorTraining;
        this.epoch = epoch;
    }

    public double getErrorMax() {
        return errorMax;
    }

    public double getErrorTraining() {
        return errorTraining;
    }

    public int getEpoch() {
        return epoch;
    }
    
    
}
