package lab2;

import java.util.*;
import java.util.stream.Collectors;

class CustomWekaImpl {

    private List<ClassificationItem> dataset;

    private double logBase;

    CustomWekaImpl(List<ClassificationItem> dataset, double logBase) {
        this.dataset = dataset;
        this.logBase = logBase;
    }

    double gainRatioAttributeEval() {
        double entrAttr = countEntropyAttribute();
        if (entrAttr == 0) return 0;
        double entrClass = countEntropyClass();
        double entrCond = countEntropyConditional();
        return (entrClass - entrCond) / entrAttr;
    }


    List<String> getValuesOfAttribute() {
        List<String> vals = dataset.stream()
                .map(ClassificationItem::getValue)
                .collect(Collectors.toList());
        vals.forEach(System.out::println);
        return vals;
    }

    List<String> getClassesOfAttribute() {
        return dataset.stream()
                .map(ClassificationItem::getGivenClass)
                .collect(Collectors.toList());
    }

    Map<String, List<String>> getBatches() {
        List<String> keys = dataset.stream().map(ClassificationItem::getValue).distinct().collect(Collectors.toList());
        Map<String, List<String>> map = new HashMap<>();
        keys.forEach(key -> {
            List<String> vals = dataset.stream().filter(item -> key.equals(item.getValue())).map(ClassificationItem::getGivenClass).collect(Collectors.toList());
            map.put(key, vals);
        });
        return map;
    }

    double countEntropyClass() {
        return new EntropyUtils<>(getClassesOfAttribute()).countEntropy();
    }

    double countEntropyAttribute() {
        return new EntropyUtils<>(getValuesOfAttribute()).countEntropy();
    }

    double countEntropyConditional() {
        int size = dataset.size();
        Map<String, List<String>> batches = getBatches();
        List<Double> entropies = new ArrayList<>();

        batches.forEach((key, vals) -> {
            double probab = (double)(vals.size()) / size;
            EntropyUtils<String> entropyUtils = new EntropyUtils<>(vals, logBase);
            double result = probab * entropyUtils.countEntropy();
            entropies.add(result);
        });
        return entropies.stream().reduce((d1, d2) -> d1+d2).orElse(0d);
    }

    private static final class EntropyUtils <T> {
        private List<T> values;
        private List<T> valuesDistinct;
        private double logBase;

        EntropyUtils(double logBase) {
            this(new ArrayList<>(), logBase);
        }

        EntropyUtils(List<T> values) {
            this(values, Math.E);
        }


        EntropyUtils(List<T> values, double logBase) {
            this.values = values;
            this.logBase = logBase;
            valuesDistinct = values.stream().distinct().collect(Collectors.toList());
        }

        double countEntropy() {
            return (-1) * valuesDistinct.stream().map(this::getSingleEntropy).reduce((d1, d2) -> d1 + d2).orElse(0d);
        }

        private double getSingleEntropy(T value) {
            double probab = getProbability(value);
            return probab * logn(0.5, probab);
        }

        private double getProbability(T value) {
            int size = values.size();
            if (size == 0) return 0;
            int valueOccurrences = (int) (values.stream().filter(d -> d.equals(value)).count());
            return (double)valueOccurrences/size;
        }

        private static double logn(double base, double value){
            return Math.log(value)/Math.log(base);
        }

        public static void main(String[] args) {

        }
    }
}
