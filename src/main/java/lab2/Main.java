package lab2;

import lab1.io.WekaReader;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static lab1.ZtadSolverMain.toArffFile;

public class Main {
    public static void main(String[] args) {
        String wekaFile = toArffFile("238454L2 2");
        List<ClassificationItem> dataset = Arrays.asList(
                new ClassificationItem(1d, "A"),
                new ClassificationItem(5d, "B"),
                new ClassificationItem(5d, "A"),
                new ClassificationItem(4d, "A"),
                new ClassificationItem(5d, "B"),
                new ClassificationItem(1d, "B"),
                new ClassificationItem(2d, "B"),
                new ClassificationItem(3d, "B"),
                new ClassificationItem(4d, "A"),
                new ClassificationItem(5d, "B")
        );

        WekaReader reader = new WekaReader(wekaFile);
        try {
            Instances elems = reader.getData();

            CustomWekaImpl wekaImpl = new CustomWekaImpl(dataset, Math.E);
            double ratio = wekaImpl.gainRatioAttributeEval("B");

            System.out.println("FINAL RATIO: " + ratio);
            System.out.println(wekaImpl.getBatches());
//            wekaImpl.abc();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
