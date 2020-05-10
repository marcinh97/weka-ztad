import io.WekaReader;
import io.WekaWriter;
import weka.core.Instances;

import static analyzer.WekaAnalyzer.filterByExpression;
import static analyzer.WekaAnalyzer.removeAttribute;

public class ZtadSolverMain {

    private static final String FILES_FOLDER = "src/main/resources/files/";

    private static String toArffFile(String fileName) {
        return FILES_FOLDER + fileName + ".arff";
    }

    public static void main(String[] args) {
        String wekaFile = toArffFile("238454L2 2");
        String wekaOutput = toArffFile("238454L3 2");
        String filter =
                "((ATT1 is 'splacona_cz' or ATT1 is 'splacona' or ATT1 is 'windykacja_sp' or ATT1 is 'windykacja') and ATT2<=900)";
        int loanStatusIndex = 0;

        WekaReader reader = new WekaReader(wekaFile);
        WekaWriter writer = new WekaWriter(wekaOutput);

        try {
            Instances elems = reader.getData();
            elems = filterByExpression(elems, filter);
            elems = removeAttribute(elems, loanStatusIndex);
            elems.forEach(System.out::println);
            writer.saveData(elems);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
