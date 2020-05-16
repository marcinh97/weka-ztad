package lab1;

import lab1.io.WekaReader;
import lab1.io.WekaWriter;
import weka.core.Instances;

import static lab1.analyzer.WekaAnalyzer.filterByExpression;
import static lab1.analyzer.WekaAnalyzer.removeAttribute;

public class ZtadSolverMain {

    private static final String FILES_FOLDER = "src/main/resources/files/";

    public static String toArffFile(String fileName) {
        return FILES_FOLDER + fileName + ".arff";
    }

    public static void main(String[] args) {
        String wekaFile = toArffFile("238454L2 2");
        String wekaOutput = toArffFile("238454L3 2");
        String filter =
                "((ATT1 is 'splacona_cz' or ATT1 is 'splacona' or ATT1 is 'windykacja_sp' or ATT1 is 'windykacja') and ATT2<=900)";
        int loanStatusIndex = 0;

        try {
            WekaReader reader = new WekaReader(wekaFile);
            Instances elems = reader.getData();
            elems = filterByExpression(elems, filter);
            elems = removeAttribute(elems, loanStatusIndex);
            System.out.println("Zapisywanie do pliku...");
            WekaWriter writer = new WekaWriter(wekaOutput);
            writer.saveData(elems);
            System.out.println("Zapisano.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
