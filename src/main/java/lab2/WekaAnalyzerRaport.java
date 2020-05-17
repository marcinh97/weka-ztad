package lab2;

import lab1.io.WekaReader;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.util.ArrayList;
import java.util.List;

import static lab1.ZtadSolverMain.toArffFile;
import static java.lang.String.format;

public class WekaAnalyzerRaport {
    private static final int[] ATTRIBUTES_TO_CHECK = {0, 1, 2, 3, 4, 5, 6, 7};

    private static String getSummary(String fileName, double logBase) throws Exception{
        List<ClassificationItem<String>> classificationItems;
        CustomWekaImpl wekaImpl;
        StringBuilder summary = new StringBuilder();
        WekaReader reader = new WekaReader(toArffFile(fileName));
        Instances elems = prepareData(reader.getData());
        for (int attrIndex : ATTRIBUTES_TO_CHECK) {
            classificationItems = prepareDataSet(elems, attrIndex);
            wekaImpl = new CustomWekaImpl<>(classificationItems, logBase);
            String attrName = elems.attribute(attrIndex).name();
            summary.append(format("%.5f for (%d): \"%s\"\n", wekaImpl.gainRatioAttributeEval(), attrIndex+1, attrName));
        }
        return summary.toString();
    }

    private static Instances prepareData(Instances instances) throws Exception {
        instances.setClassIndex(instances.numAttributes()-1);
        Discretize discretize = new Discretize();
        discretize.setInputFormat(instances);
        discretize.setOptions(new String[]{"-R", "1,2,3,5,6,7,8"});
        instances = Filter.useFilter(instances, discretize);

        NumericToNominal numToNominalFilter = new NumericToNominal();
        numToNominalFilter.setInputFormat(instances);
        numToNominalFilter.setOptions(new String[]{"-R", "4"});
        return Filter.useFilter(instances, numToNominalFilter);
    }

    private static List<ClassificationItem<String>> prepareDataSet(Instances instances, int attrNum) throws Exception {
        List<ClassificationItem<String>> items = new ArrayList<>();
        Instances discretizedData = prepareData(instances);
        int classAttrIndex = instances.numAttributes()-1;
        for (Instance instance : discretizedData) {
            String value = instance.stringValue(attrNum);
            String classification = instance.stringValue(classAttrIndex);
            items.add(new ClassificationItem<>(value, classification));
        }
        return items;
    }

    public static void main(String[] args) {
        String fileName = "238454L3 1";
        double logBase = Math.E;
        try {
            System.out.println(getSummary(fileName, logBase));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
