import org.apache.commons.io.FileUtils;
import org.joda.time.DateTime;

import java.io.*;
import java.util.HashMap;

/**
 * Created by cxa123230 on 10/2/2017.
 * takes txinall and txoutall files from the output of GraphParser and finds graphlets per the used time period (e.g., week)
 */


public class ChainletDiscoverer {
    static int timePeriodMax = 365;

    public static void main(String[] args) throws Exception {
        String dir = "D:\\Bitcoin\\createddata\\daily\\";
        String crdir = "D:\\Bitcoin\\createddata\\dailyOccMatrices\\";
        FileUtils.cleanDirectory(new File(crdir));
        for (int year = 2009; year <= 2017; year++) {

            for (int timePeriod = 1; timePeriod <= timePeriodMax; timePeriod++) {
                int dim = 20;
                int[][] occM = new int[dim][dim];
                String fileName = dir + year + "_" + timePeriod + ".txt";
                File f = new File(fileName);
                if (!f.exists()) {
                    System.out.println(fileName + " does not exist");
                    continue;
                }
                BufferedReader inBr = new BufferedReader(new FileReader(fileName));
                HashMap<String, Integer> inTranx = new HashMap<String, Integer>();
                HashMap<String, Integer> outTranx = new HashMap<String, Integer>();
                loadTranx(inBr, year, timePeriod, inTranx, outTranx);
                inBr.close();

                int sSec = outTranx.size();

                int transition = 0;
                int merge = 0;
                int split = 0;
                for (String txId : inTranx.keySet()) {
                    //merge, split, transition
                    int icount = inTranx.get(txId);
                    if (outTranx.containsKey(txId)) {

                        int ocount = outTranx.get(txId);
                        if (icount == ocount) transition++;
                        else if (icount > ocount) merge++;
                        else split++;
                        if (icount > dim) icount = dim;
                        if (ocount > dim) ocount = dim;
                        occM[icount - 1][ocount - 1]++;

                    }
                }
                double total = merge + split + transition;
                System.out.println(year + "\t" + timePeriod + "\t" + sSec + "\t" + merge / total + "\t" + split / total + "\t" + transition / total);
                if (total > 0) {
                    writeMatrix(year, timePeriod, occM, crdir, "occ");
                }
            }
        }
    }


    private static void writeMatrix(int year, int timePeriod, int[][] occ, String dir, String file) throws Exception {
        String x = "";
        if (timePeriodMax == 365) x = "day";
        else if (timePeriodMax == 52) x = "week";
        else {
            throw new RuntimeException("time period is unknown. Should be day or week.");
        }
        BufferedWriter wr = new BufferedWriter(new FileWriter(dir + file + year + x + timePeriod + ".csv"));
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

    private static void loadTranx(BufferedReader inBr, int year, int week, HashMap<String, Integer> inTranx, HashMap<String, Integer> outTranx) throws IOException {
        String line;
        int i = 0;
        while ((line = inBr.readLine()) != null) {
            String[] arr = line.split("\t");
            DateTime time = new DateTime(1000 * Long.parseLong(arr[1]));
            int txYear = time.year().get();
            if (year == txYear) {
                String txId = (arr[0]);
                int txTimePeriod = 0;
                txTimePeriod = CheckTxTimePeriod(time);
                if (week == txTimePeriod) {
                    int ins = Integer.parseInt(arr[2]);
                    int outs = Integer.parseInt(arr[3]);
                    inTranx.put(txId, ins);
                    outTranx.put(txId, outs);

                }
            }
        }

    }

    private static int CheckTxTimePeriod(DateTime time) {
        int txTimePeriod;
        if (timePeriodMax == 52) txTimePeriod = time.weekOfWeekyear().get();
        else if (timePeriodMax == 365) txTimePeriod = time.getDayOfYear();
        else {
            throw new RuntimeException("time period is unknown. Should be day or week.");
        }
        return txTimePeriod;
    }
}
