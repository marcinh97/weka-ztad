package lab3;

import static lab3.Validator.SingleValidationResult;

import lombok.Data;

@Data
class ConfusionMatrix {
    private double positiveValueLabel;
    private int truePositives;
    private int falsePositives;
    private int falseNegatives;
    private int trueNegatives;

    ConfusionMatrix(double positiveValueLabel) {
        this(positiveValueLabel, 0, 0, 0, 0);
    }

    ConfusionMatrix(double positiveValueLabel,
                    int truePositives, int falsePositives, int falseNegatives, int trueNegatives) {
        this.positiveValueLabel = positiveValueLabel;
        this.truePositives = truePositives;
        this.falsePositives = falsePositives;
        this.falseNegatives = falseNegatives;
        this.trueNegatives = trueNegatives;
    }

    void update(SingleValidationResult validationResult) {
        truePositives += isTruePositive(validationResult) ? 1 : 0;
        falseNegatives += isFalseNegative(validationResult) ? 1 : 0;
        falsePositives += isFalsePositive(validationResult) ? 1 : 0;
        trueNegatives += isTrueNegative(validationResult) ? 1 : 0;
    }

    void average(int num) {
        if (num == 0) return;
        truePositives /= num;
        falseNegatives /= num;
        falsePositives /= num;
        trueNegatives /= num;
    }

    private boolean isTruePositive(SingleValidationResult validationResult) {
        return isPositive(validationResult.getRealValue()) && isPositive(validationResult.getClassifierValue());
    }

    private boolean isFalseNegative(SingleValidationResult validationResult) {
        return isPositive(validationResult.getRealValue()) && !isPositive(validationResult.getClassifierValue());
    }

    private boolean isFalsePositive(SingleValidationResult validationResult) {
        return !isPositive(validationResult.getRealValue()) && isPositive(validationResult.getClassifierValue());
    }

    private boolean isTrueNegative(SingleValidationResult validationResult) {
        return !isPositive(validationResult.getRealValue()) && !isPositive(validationResult.getClassifierValue());
    }

    private boolean isPositive(double value) {
        return value == positiveValueLabel;
    }

    double calculateAccuracy() {
        return (double) (truePositives + trueNegatives)
                / (truePositives + trueNegatives + falseNegatives + falsePositives);
    }

    double calculateTNrate() {
        return (double) trueNegatives / (trueNegatives + falsePositives);
    }

    double calculateTPrate() {
        return (double) truePositives / (truePositives + falseNegatives);
    }

    private double calculateFPrate() {
        return (double) falsePositives / (falsePositives + trueNegatives);
    }

    double calculateGMean() {
        return Math.sqrt(calculateTPrate() * calculateTNrate());
    }

    double calculateAUC() {
        return (1 + calculateTPrate() - calculateFPrate()) / 2;
    }

    @Override
    public String toString() {
        return "TP: " + truePositives +
                ", FP: " + falsePositives +
                ", FN: " + falseNegatives +
                ", TN: " + trueNegatives;
    }
}