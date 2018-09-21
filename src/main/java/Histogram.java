import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.TreeMap;

public class Histogram {
    private static final Logger logger = LoggerFactory.getLogger(Histogram.class);
    static long x[] = new long[]{1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 200000, 400000, 800000, 1000000, 10000000, 100000000, 1000000000, 10000000000L, 100000000000L, 1000000000000L, 10000000000000L, 100000000000000L, 1000000000000000L};

    public static void main(String args[]) throws IOException {
        String dataDir = args[0];
        String out = args[1];
        File folder = new File(dataDir);
        File[] listOfFiles = folder.listFiles();
        FileUtils.cleanDirectory(new File(out));
        BufferedWriter wr = new BufferedWriter(new FileWriter(out + "dist.csv", true));
        wr.write("year\tmonth\tbalance\tfrequency\r\n");
        for (int year = 2009; year < 2018; year++)
            for (int moth = 1; moth <= 12; moth++) {
                String c = year + "_" + moth + ".txt";
                TreeMap<Long, Long> amountMap = new TreeMap<>();
                for (File dirFile : listOfFiles) {
                    BufferedReader br = new BufferedReader(new FileReader(dirFile));
                    String name = dirFile.getName();
                    String fName = name.substring(4 + name.indexOf("dist"));
                    if (!fName.equalsIgnoreCase(c)) {
                        continue;
                    }
                    logger.info(name + " is processed for " + c);
                    String line = "";
                    while ((line = br.readLine()) != null) {
                        String arr[] = line.split("\t");
                        long val = cut(Long.parseLong(arr[0]));
                        long count = Long.parseLong(arr[1]);

                        if (!amountMap.containsKey(val)) amountMap.put(val, count);
                        else amountMap.put(val, count + amountMap.get(val));
                    }
                }


                for (Long balance : amountMap.keySet()) {

                    wr.write(year + "\t" + moth + "\t" + balance + "\t" + amountMap.get(balance) + "\r\n");
                }

            }


        wr.close();
    }

    private static long cut(long b) {


        for (int i = 0; i < x.length; i++) {
            if (b <= x[i]) {
                return x[i];
            }
        }
        return x[x.length - 1];

    }
}
