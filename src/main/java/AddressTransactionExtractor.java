import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

public class AddressTransactionExtractor {
    /*
    Links transaction outputs and inputs to determine when an address spends its coins.
     */
    private static final Logger logger = LoggerFactory.getLogger(AddressTransactionExtractor.class);

    public static void main(String args[]) throws IOException {
        String dataDir = args[0];
        String outDir = args[1];
        BufferedReader br;
        FileUtils.cleanDirectory(new File(outDir));


        for (int year = 2009; year < 2018; year++) {
            for (int month = 1; month <= 12; month++) {

                Map<String, String[]> aMap = new HashMap<String, String[]>(200000);

                br = new BufferedReader(new FileReader(dataDir + "outputs" + year + "_" + month + ".txt"));
                String line = "";
                while ((line = br.readLine()) != null) {
                    try {
                        String[] split = line.split("\t");
                        String txId = split[1];
                        int length = split.length;
                        for (int i = 2; i < length; i = i + 2) {
                            //new output address
                            String address = split[i];
                            if (address.equalsIgnoreCase("noaddress")) continue;
                            String amount = (split[i + 1]);
                            String key = txId + "_" + (i - 2);
                            aMap.put(key, new String[]{address, amount, split[0]});
                        }
                    } catch (Exception e) {
                        logger.error("error line " + line);
                    }
                }
                logger.info(year + " " + month + " " + aMap.size() + " transactions");
                int y2 = year;
                int m2 = month;
                for (; y2 < 2018; y2++) {
                    for (; m2 <= 12; m2++) {
                        if (y2 == 2017 && m2 >= 9) continue;
                        logger.info(">>" + y2 + " " + m2);
                        StringBuilder bf = new StringBuilder();
                        BufferedReader br2 = new BufferedReader(new FileReader(dataDir + "inputs" + y2 + "_" + m2 + ".txt"));
                        BufferedWriter wr = new BufferedWriter(new FileWriter(outDir + "inputs" + y2 + "_" + m2 + ".txt", true));
                        String l2 = "";
                        while ((l2 = br2.readLine()) != null) {
                            String[] split = l2.split("\t");
                            int length = split.length;
                            if (length % 2 != 0) continue;
                            for (int i = 2; i < length; i = i + 2) {
                                String prevTx = split[i];
                                String index = split[i + 1];
                                String key = prevTx + "_" + (index);
                                if (aMap.containsKey(key)) {
                                    String[] comArr = aMap.get(key);
                                    String addres = comArr[0];
                                    String value = comArr[1];
                                    String earnedTime = comArr[2];
                                    String spentTime = split[0];
                                    bf.append(addres);
                                    bf.append("\t");
                                    bf.append(value);
                                    bf.append("\t");
                                    bf.append(earnedTime);
                                    bf.append("\t");
                                    bf.append(spentTime);
                                    bf.append("\r\n");
                                    aMap.remove(key);
                                }
                            }
                        }
                        wr.append(bf.toString());
                        wr.close();
                        br2.close();
                    }
                    m2 = 1;
                }
                BufferedWriter wr2 = new BufferedWriter(new FileWriter(outDir + "unspent" + year + "_" + month + ".txt", true));
                StringBuilder buf = new StringBuilder();
                for (String key : aMap.keySet()) {
                    String[] comArr = aMap.get(key);
                    String address = comArr[0];
                    String amount = comArr[1];
                    String time = comArr[2];
                    buf.append(address + "\t" + time + "\t" + amount + "\n");
                }
                wr2.append(buf.toString());
                wr2.close();
            }
        }
    }
}
