package Anomaly;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class SortFile {
    private static final Logger logger = LoggerFactory.getLogger(SortFile.class);

    public static void main(String args[]) throws IOException {
        String dataDir = args[0];
        String outDir = args[1];
        String line = "";
        FileUtils.cleanDirectory(new File(outDir));
        SimpleDateFormat sdf = new SimpleDateFormat("EEEE,MMMM d,yyyy h:mm,a");
        sdf.setTimeZone(TimeZone.getTimeZone("UTC"));

        splitIntoDays(dataDir, "inputs", outDir);
        splitIntoDays(dataDir, "outputs", outDir);
    }

    private static void splitIntoDays(String dataDir, String type, String outDir) throws IOException {
        String line;
        for (int year = 2009; year < 2018; year++) {
            for (int month = 1; month <= 12; month++) {
                logger.info(year + "\t" + month);
                HashMap<Integer, List<String>> con = new HashMap<>();
                String fPath = dataDir + type + year + "_" + month + ".txt";
                if (!new File(fPath).exists()) continue;
                BufferedReader br = new BufferedReader(new FileReader(fPath));

                while ((line = br.readLine()) != null) {
                    String arr[] = line.split("\t");
                    long txTime = Long.parseLong(arr[0]) * 1000;
                    Date date = new Date(txTime);
                    Calendar cal = Calendar.getInstance();
                    cal.setTime(date);
                    int day = cal.get(Calendar.DAY_OF_YEAR);

                    if (!con.containsKey(day)) {
                        con.put(day, new ArrayList<String>());
                    }
                    con.get(day).add(line);
                    if (con.get(day).size() > 5000) {
                        writeContent(type, year, con, day, outDir);
                        con.get(day).clear();
                    }
                }
                br.close();
                for (int day : con.keySet()) {
                    if (con.get(day).size() > 0) {
                        writeContent(type, year, con, day, outDir);
                    }
                }
            }
        }
    }

    private static void writeContent(String type, int year, HashMap<Integer, List<String>> con, int day, String outDir) throws IOException {
        BufferedWriter wr = new BufferedWriter(new FileWriter(outDir + type + year + "Day" + day + ".txt", true));
        for (String s : con.get(day)) {
            wr.append(s + "\r\n");
        }
        wr.close();
    }
}