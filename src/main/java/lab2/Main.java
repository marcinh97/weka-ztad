package lab2;

import lab1.io.WekaReader;
import labxx.CustomNumericToNominalProcessor;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.util.ArrayList;
import java.util.List;

import static lab1.ZtadSolverMain.toArffFile;

public class Main {

    static Instances prepareData(Instances instances) throws Exception {
        Discretize discretize = new Discretize();
        discretize.setInputFormat(instances);
        String[] options = {"-R", "1,2,3,4,5,6,7,8,9"};
        discretize.setOptions(options);
        return Filter.useFilter(instances, discretize);
    }

    static List<ClassificationItem> prepareDataSet(Instances instances, int attrNum, int classAttrIndex) throws Exception {
        List<ClassificationItem> items = new ArrayList<>();
        Instances discretizedData = prepareData(instances);
        discretizedData.forEach(System.out::println);

        for (Instance instance : discretizedData) {
            String  value = instance.stringValue(attrNum);
            String classification = instance.stringValue(classAttrIndex);
            items.add(new ClassificationItem(value, classification));
        }
        return items;
    }

    public static void main(String[] args) {
        String wekaFile = toArffFile("238454L3 1");

        WekaReader reader = new WekaReader(wekaFile);

        final int classAttrIndex = 8;
        final int attrIndex = 4;
        try {
            Instances elems = reader.getData();
            elems = new CustomNumericToNominalProcessor().prepareData(elems);
            elems.setClassIndex(classAttrIndex); // status pozyczki - atrybut klasy dobry/zly
            GainRatioAttributeEval gainRatioAttributeEval = new GainRatioAttributeEval();
            gainRatioAttributeEval.buildEvaluator(elems);
            System.out.println(gainRatioAttributeEval.evaluateAttribute(attrIndex)); // okres w jakim pobral, wartosc 0.068

            List<ClassificationItem> classificationItems = prepareDataSet(elems, attrIndex, classAttrIndex);

            CustomWekaImpl wekaImpl = new CustomWekaImpl(classificationItems, 0.5);
//            wekaImpl.gainRatioAttributeEval()
//            System.out.println(wekaImpl.countEntropyAttribute());
            System.out.println("For "+attrIndex+" final: " + wekaImpl.gainRatioAttributeEval());

//
//            System.out.println("FINAL RATIO: " + ratio);
            System.out.println(wekaImpl.getBatches());

//            wekaImpl.abc();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
