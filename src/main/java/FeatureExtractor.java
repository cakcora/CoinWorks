import org.apache.commons.io.FileUtils;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.joda.time.DateTime;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by cxa123230 on 2/10/2018.
 */
public class FeatureExtractor {

    public static void main(String[] args) throws Exception {
        boolean preprocess = false;
        String dir = "D:\\Bitcoin/createddata/feature/";
        File file = new File("D://bitcoin/createddata/feature");
        file.mkdirs();
        FileUtils.cleanDirectory(file);
        int years[] = {2016, 2017};

        for (int year : years) {
            splitFiles(year);
        }
        System.out.println("year\tday\tmeanValue\tmedianValue\thoMedian\tmeanDegree\tmedianDegree\taddCount\ttxCount");
        for (int year : years) {
            for (int day = 1; day <= 365; day++) {
                BufferedReader br = new BufferedReader(new FileReader(dir + year + "_" + day + ".txt"));
                String line = "";
                DescriptiveStatistics amounts = new DescriptiveStatistics();
                HashMap<Integer, Integer> hourlyTx = new HashMap<>();
                HashMap<String, Integer> inDegrees = new HashMap<>();
                HashMap<String, Integer> outDegrees = new HashMap<>();
                HashSet<String> addresses = new HashSet<>();
                while ((line = br.readLine()) != null) {
                    String arr[] = line.split("\t");
                    String prefix = arr[0];
                    String tx = arr[2];
                    // medianValue of the hourly transactions
                    DateTime blockDate = new DateTime(1000 * Long.parseLong(arr[1]));
                    int thishour = blockDate.getHourOfDay();
                    if (!hourlyTx.containsKey(thishour)) hourlyTx.put(thishour, 0);
                    hourlyTx.put(thishour, hourlyTx.get(thishour) + 1);

                    if (prefix.equalsIgnoreCase("i")) {
                        inDegrees.put(tx, (arr.length - 3) / 2);

                    } else if (prefix.equalsIgnoreCase("o")) {

                        outDegrees.put(tx, (arr.length - 3) / 2);

                        long amount = 0;
                        for (int i = 3; i <= arr.length - 2; i = i + 2) {
                            amount += Long.parseLong(arr[i + 1]);
                            addresses.add(arr[i]);
                        }
                        amounts.addValue(amount);
                    }
                }
                //meanValue transaction value
                double meanValue = amounts.getMean();
                //medianValue transaction value
                double medianValue = amounts.getPercentile(0.5);
                //median of hourly transaction count
                DescriptiveStatistics hoTx = new DescriptiveStatistics();
                for (int v : hourlyTx.values()) {
                    hoTx.addValue(v);
                }
                double hoMedian = hoTx.getPercentile(0.5);
                //median degree
                DescriptiveStatistics degrees = new DescriptiveStatistics();
                for (String tx : inDegrees.keySet()) {
                    if (outDegrees.containsKey(tx)) {
                        Integer degree = inDegrees.get(tx);
                        for (int f = 1; f < outDegrees.get(tx); f++) {
                            degrees.addValue(degree);
                        }
                    }
                }
                double meanDegree = degrees.getMean();
                double medianDegree = degrees.getPercentile(0.5);
//                    number of new addresses
                int addCount = addresses.size();
                // transaction count
                int txCount = inDegrees.size();
                System.out.println(year + "\t" + day + "\t" + meanValue + "\t" + medianValue + "\t" + hoMedian + "\t" + meanDegree + "\t" + medianDegree + "\t" + addCount + "\t" + txCount);
            }
        }
    }
    private static void splitFiles(int refYear) throws Exception {


        HashMap<Long, StringBuffer> content = new HashMap<>();
        String f[] = {"D://bitcoin/createddata/txInputs.txt", "D://bitcoin/createddata/txOutputs.txt"};
        for (String fileName : f) {

            String substring = fileName.substring(26, 27);
            BufferedReader inBr = new BufferedReader(new FileReader(fileName));
            HashSet<String> txIds = new HashSet<>();
            String line = "";
            int l = 0;
            while ((line = inBr.readLine()) != null) {
                if (++l % 1000000 == 0) {
                    System.out.println(l);
                }
                if (line.length() < 10) continue;
                String arr[] = line.split("\t");

                DateTime blockDate = new DateTime(1000 * Long.parseLong(arr[0]));
                long year = blockDate.getYear();
                if (year == refYear) {
                    String tx = arr[1];
                    long day = blockDate.getDayOfYear();
                    if (!content.containsKey(day)) content.put(day, new StringBuffer());
                    content.get(day).append(substring + "\t" + line + "\r\n");
                    if (content.get(day).length() > 100000) {
                        write(year, day, content.get(day));
                        content.remove(day);
                    }
                }
            }
        }
        for (Long c : content.keySet()) {
            write(refYear, c, content.get(c));
        }
    }

    private static void write(long year, long day, StringBuffer stringBuffer) throws Exception {
        new BufferedWriter(new FileWriter("D://bitcoin/createddata/feature/" + year + "_" + day + ".txt", true)).append(stringBuffer.toString()).close();
    }

}
