import org.apache.commons.io.FileUtils;

import java.io.*;
import java.util.Arrays;

/**
 * Created by cxa123230 on 10/2/2017.
 * takes txinall and txoutall files from the output of GraphParser and finds graphlets per the used time period (e.g., week)
 */


public class ChainletMatrix {
    static int timePeriodMax = 366;

    public static void main(String[] args) throws Exception {
        String[] coins = {("Litecoin"), ("Namecoin"), ("Bitcoin")};
        for (String coin : coins) {
            getmatrix(coin);
        }
    }

    private static void getmatrix(String coin) throws Exception {
        String dir = "D:\\" + coin + "/createddata/daily/";
        String occDir = "D:\\" + coin + "/createddata/dailyOccMatrices/";
        String amoDir = "D:\\" + coin + "/createddata/dailyAmoMatrices/";
        checkTxTimePeriod(timePeriodMax);

        File d1 = new File(occDir);
        d1.mkdirs();
        FileUtils.cleanDirectory(d1);
        File d2 = new File(amoDir);
        d2.mkdirs();
        FileUtils.cleanDirectory(d2);
        BufferedWriter wrinfo = new BufferedWriter(new FileWriter(dir + "info.txt"));

        for (int year = 2009; year <= 2017; year++) {

            for (int timePeriod = 1; timePeriod <= timePeriodMax; timePeriod++) {
                int dim = 20;
                Integer[][] occM = new Integer[dim][dim];
                Long[][] amoM = new Long[dim][dim];
                for (Integer[] row : occM)
                    Arrays.fill(row, 0);
                for (Long[] row : amoM)
                    Arrays.fill(row, 0L);
                String fileName = dir + year + "_" + timePeriod + ".txt";
                File f = new File(fileName);
                if (!f.exists()) {
                    System.out.println(fileName + " does not exist");
                    continue;
                }
                BufferedReader inBr = new BufferedReader(new FileReader(fileName));
                int i = 0;
                String line = "";
                int transition = 0;
                int merge = 0;
                int split = 0;
                while ((line = inBr.readLine()) != null) {
                    String[] arr = line.split("\t");
                    int icount = Integer.parseInt(arr[2]);
                    int ocount = Integer.parseInt(arr[3]);
                    if (icount == ocount) transition++;
                    else if (icount > ocount) merge++;
                    else split++;
                    if (icount > dim) icount = dim;
                    if (ocount > dim) ocount = dim;
                    occM[icount - 1][ocount - 1]++;
                    amoM[icount - 1][ocount - 1] += Long.parseLong(arr[4]);

                }
                inBr.close();

                double total = merge + split + transition;
                wrinfo.append(year + "\t" + timePeriod + "\t" + (long) total + "\t" + merge / total + "\t" + split / total + "\t" + transition / total + "\r\n");
                if (total > 0) {
                    writeMatrix(year, timePeriod, occM, occDir, "occ");
                    writeMatrix(year, timePeriod, amoM, amoDir, "amo");
                }
            }
        }
        wrinfo.close();
    }


    private static <T extends Number> void writeMatrix(int year, int periodIndex, T[][] occ, String dir, String file) throws Exception {
        String x = "";
        String updatedFileName = periodIndex + "";
        if (updatedFileName.length() == "1".length()) updatedFileName = "00" + updatedFileName;
        else if (updatedFileName.length() == "11".length()) updatedFileName = "0" + updatedFileName;
        BufferedWriter wr = new BufferedWriter(new FileWriter(dir + file + year + x + updatedFileName + ".csv"));
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


    private static void checkTxTimePeriod(int txTimePeriod) {
        if (timePeriodMax != 52 && timePeriodMax != 366) {
            throw new RuntimeException("time period is unknown. Should be day or week.");
        }
        return;
    }
}
