package Price;

import org.apache.commons.io.FileUtils;
import org.joda.time.DateTime;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeSet;

/**
 * Created by cxa123230 on 3/28/2018.
 */
public class EdgeNetworkSplitter {

    public static void main(String[] args) throws Exception {
        String[] coins = {("Bitcoin")};
        for (String coin : coins) {
            //getmatrix(coin);
        }
        writeMap("bitcoin");
    }

    private static void writeMap(String coin) throws IOException {
        String dir = "D:\\" + coin + "/createddata/";
        String edgeDir = dir + "edges/";
        String line;
        int f = 0;

        String ch = "a";
        char[] chs = new char[]{'a', 'b', 'c', 'd', 'e', 'f', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

        for (char searchChar : chs) {
            Map<String, String> maps = new HashMap<>();

            BufferedReader inBr = new BufferedReader(new FileReader(dir + "txInputs.txt"));
            while ((line = inBr.readLine()) != null) {
                try {
                    String[] arr = line.split("\t");
                    String txId = arr[1];

                    if (txId.charAt(0) == searchChar) {
                        DateTime time = new DateTime(1000 * Long.parseLong(arr[0]));
                        int txYear = time.year().get();
                        int txMonth = time.monthOfYear().get();
                        String key = txYear + "_" + txMonth;

                        maps.put(txId, key);
                        if (f++ % 100000 == 0) System.out.println(f + " " + key);
                    }
                } catch (Exception e) {
                    System.out.println(line + " " + e.getMessage());
                }
            }
            BufferedWriter wr = new BufferedWriter(new FileWriter(edgeDir + searchChar + "map.txt", true));
            TreeSet<String> sorted = new TreeSet<String>(maps.keySet());
            for (String tx : sorted) {
                wr.append(tx + "\t" + maps.get(tx) + "\r\n");
            }
            wr.close();
        }

    }

    private static void getmatrix(String coin) throws IOException {
        String dir = "D:\\" + coin + "/createddata/";
        String eDir = dir + "edges/";
        String inFIle = "txInputs.txt";
        String outFIle = "txOutputs.txt";
        File d1 = new File(eDir);
        d1.mkdirs();
        FileUtils.cleanDirectory(d1);
        BufferedReader inBr = new BufferedReader(new FileReader(dir + "txInputs.txt"));
        BufferedReader outBr = new BufferedReader(new FileReader(dir + "txOutputs.txt"));

        processFile(eDir, outBr, "outputs");
        processFile(eDir, inBr, "inputs");


    }

    private static void processFile(String eDir, BufferedReader inBr, String descr) throws IOException {
        String line;
        HashMap<String, HashSet<String>> maps = new HashMap<>();
        HashMap<String, Integer> counts = new HashMap<>();
        while ((line = inBr.readLine()) != null) {
            try {
                String[] arr = line.split("\t");
                String txId = arr[1];
                if (txId.equalsIgnoreCase("264299886446921c89e598ec2b1ec3eab6a2c9b0235b310ff513a039315ff721"))
                    System.out.println("***************" + line);
                DateTime time = new DateTime(1000 * Long.parseLong(arr[0]));
                int txYear = time.year().get();
                int txMonth = time.monthOfYear().get();

                String key = txYear + "_" + txMonth;
                if (!maps.containsKey(key)) {
                    maps.put(key, new HashSet<>());
                    counts.put(key, 0);
                }
                maps.get(key).add(line);
                counts.put(key, 1 + counts.get(key));

                if (maps.get(key).size() > 50000) {
                    BufferedWriter wr = new BufferedWriter(new FileWriter(eDir + descr + key + ".txt", true));
                    for (String l : maps.get(key)) {
                        wr.append(l + "\r\n");
                    }
                    wr.close();
                    maps.remove(key);
                }
            } catch (Exception e) {
                System.out.println(line + " " + e.getMessage());
            }
        }
        for (String key : maps.keySet()) {
            BufferedWriter wr = new BufferedWriter(new FileWriter(eDir + descr + key + ".txt", true));
            for (String l : maps.get(key)) {
                wr.append(l + "\r\n");
            }
            wr.close();
        }
        for (String key : counts.keySet()) {
            System.out.print(key + "\t" + counts.get(key) + "\r\n");
        }
    }


}


