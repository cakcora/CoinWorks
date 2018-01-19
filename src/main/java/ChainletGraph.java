import org.apache.commons.io.FileUtils;

import java.io.*;
import java.util.HashMap;
import java.util.TreeMap;

/**
 * Created by cxa123230 on 10/2/2017.
 * takes txinall and txoutall files from the output of GraphParser and finds graphlets per the used time period (e.g., week)
 */


public class ChainletGraph {
    static int timePeriodMax = 366;

    public static void main(String[] args) throws Exception {
        String[] coins = {("Bitcoin"), ("Namecoin"), ("Litecoin")};
        for (String coin : coins) {
            getmatrix(coin);
        }
    }

    private static void getmatrix(String coin) throws Exception {
        String dir = "D:\\" + coin + "/createddata/daily/";
        String chDir = "D:\\" + coin + "/createddata/ChainletGraph/";


        File d1 = new File(chDir);
        d1.mkdirs();
        FileUtils.cleanDirectory(d1);


        long max = 0000;

        for (int year = 2017; year <= 2017; year++) {

            for (int timePeriod = 1; timePeriod <= timePeriodMax; timePeriod++) {
                int dim = 20;
                HashMap<String, TreeMap<Long, Integer>> grMat = new HashMap<String, TreeMap<Long, Integer>>();

                String fileName = dir + year + "_" + timePeriod + ".txt";
                File f = new File(fileName);
                if (!f.exists()) {
                    System.out.println(fileName + " does not exist");
                    continue;
                }
                BufferedReader inBr = new BufferedReader(new FileReader(fileName));
                String line = "";
                while ((line = inBr.readLine()) != null) {
                    String[] arr = line.split("\t");
                    int icount = Integer.parseInt(arr[2]);
                    int ocount = Integer.parseInt(arr[3]);
                    if (icount > dim) icount = dim;
                    if (ocount > dim) ocount = dim;
                    String chId = "X" + icount + ":" + ocount;
                    if (!grMat.containsKey(chId)) {
                        grMat.put(chId, new TreeMap<>());
                    }
                    long amount = Long.parseLong(arr[4]);
                    if (amount > max) {
                        max = amount;
                        System.out.println("new max:" + max);
                    }

                    TreeMap<Long, Integer> chAmounts = grMat.get(chId);
                    if (!chAmounts.containsKey(amount)) {
                        chAmounts.put(amount, 0);
                    }
                    chAmounts.put(amount, chAmounts.get(amount) + 1);
                }
                inBr.close();


                writeMatrix(year, timePeriod, chDir, grMat);

            }
        }
    }

    private static void writeMatrix(int year, int timePeriod, String dir, HashMap<String, TreeMap<Long, Integer>> grMat) throws IOException {
        String x = "";
        BufferedWriter wr = new BufferedWriter(new FileWriter(dir + year + x + timePeriod + ".csv"));
        StringBuffer bf = new StringBuffer();
        for (int i = 1; i <= 20; i++) {
            for (int o = 1; o <= 20; o++) {
                String chId = "X" + i + ":" + o;
                if (grMat.containsKey(chId)) {
                    bf.append(chId);
                    for (Long l : grMat.get(chId).keySet()) {
                        bf.append("\t" + l + ":" + grMat.get(chId).get(l));
                    }
                } else bf.append(chId + "\t" + "NULL");
                bf.append("\r\n");
            }

        }
        wr.append(bf.toString());
        wr.close();
    }


}
