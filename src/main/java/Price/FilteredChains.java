package Price;

import org.apache.commons.io.FileUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by cxa123230 on 10/2/2017.
 * takes txinall and txoutall files from the output of GraphParser and finds graphlets per the used time period (e.g., week)
 */


public class FilteredChains {
    static int timePeriodMax = 366;

    public static void main(String[] args) throws Exception {
        String[] coins = {("Bitcoin"), ("Litecoin"), ("Namecoin")};
        for (String coin : coins) {
            System.out.println(coin + " processing.");
            getmatrix(coin);
        }
    }

    private static void getmatrix(String coin) throws Exception {
        String dir = "D:\\" + coin + "/createddata/daily/";
        String filterDir = "D:\\" + coin + "/createddata/filteredDailyOccMatrices/";

        File d2 = new File(filterDir);
        d2.mkdirs();
        FileUtils.cleanDirectory(d2);

        for (int year = 2009; year <= 2017; year++) {

            for (int timePeriod = 1; timePeriod <= timePeriodMax; timePeriod++) {
                String fileName = dir + year + "_" + timePeriod + ".txt";
                File f = new File(fileName);
                if (!f.exists()) {
                    System.out.println(fileName + " does not exist");
                    continue;
                }
                BufferedReader inBr = new BufferedReader(new FileReader(fileName));
                String line = "";
                ArrayList<String> content = new ArrayList<String>();
                while ((line = inBr.readLine()) != null) {
                    content.add(line);
                }
                inBr.close();
                for (int filterIndex = 0; filterIndex < 100; filterIndex += 10) {
                    int dim = 20;
                    boolean has = false;
                    double threshold = (double) filterIndex;
                    Integer[][] occM = new Integer[dim][dim];

                    for (Integer[] row : occM)
                        Arrays.fill(row, 0);

                    for (String l : content) {
                        String[] arr = l.split("\t");
                        int icount = Integer.parseInt(arr[2]);
                        int ocount = Integer.parseInt(arr[3]);
                        double amount = Double.parseDouble(arr[4]) / Math.pow(10, 8);
                        if (amount > threshold) {
                            if (icount > dim) icount = dim;
                            if (ocount > dim) ocount = dim;
                            occM[icount - 1][ocount - 1]++;
                            has = true;
                        }
                    }
                    if (has)
                        writeMatrix(year, timePeriod, occM, filterDir, "occ", filterIndex);
                }

            }
        }
    }


    private static <T extends Number> void writeMatrix(int year, int periodIndex, T[][] occ, String dir, String file, int filterIndex) throws Exception {
        String updatedFileName = periodIndex + "";

        BufferedWriter wr = new BufferedWriter(new FileWriter(dir + file + year + "_" + updatedFileName + "_" + filterIndex + ".csv"));
        StringBuffer bf = new StringBuffer();
        for (int i = 0; i < occ.length; i++) {
            for (int j = 0; j < occ.length - 1; j++) {
                bf.append(occ[i][j] + ",");
            }
            bf.append(occ[i][occ.length - 1]);
            bf.append("\r\n");
        }
        wr.append(bf.toString());
        wr.close();
    }


}
