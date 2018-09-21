import org.joda.time.DateTime;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.math.BigInteger;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class AddressBalanceSummarizer {
    private static final Logger logger = LoggerFactory.getLogger(AddressBalanceSummarizer.class);

    public static void main(String args[]) throws IOException {

        String dataDir = args[0];
        String tempDir = "D:\\Bitcoin\\createddata\\address balance/";
        BufferedReader spentReader;
        BufferedReader unspentReader;
        Map<String, HashMap<Integer, BigInteger>> aMap = new HashMap<String, HashMap<Integer, BigInteger>>(200000);
        int addressCount = 0;
        long useless = 0, useful = 0;
        HashMap<String, Integer> addAppearance = new HashMap<>();
        BufferedReader spentReader1;
        BufferedReader unspentReader1;

        HashMap<String, Long> addBins = new HashMap<>();
        int binWidth = 2;
        for (int year1 = 2009; year1 < 2018; year1++) {
            for (int month1 = 1; month1 <= 12; month1++) {
                if (year1 == 2017 && month1 > 7) continue;
                spentReader1 = new BufferedReader(new FileReader(dataDir + "inputs" + year1 + "_" + month1 + ".txt"));
                unspentReader1 = new BufferedReader(new FileReader(dataDir + "unspent" + year1 + "_" + month1 + ".txt"));
                logger.info("Prefix strings in " + year1 + " " + month1);
                String line1 = "";
                while ((line1 = spentReader1.readLine()) != null) {
                    String a1[] = line1.split("\t");
                    String add1 = a1[0];
                    String substring = add1.substring(0, binWidth);
                    if (!addBins.containsKey(substring))
                        addBins.put(substring, 1l);
                    else addBins.put(substring, 1l + addBins.get(substring));

                }
                spentReader1.close();
                while ((line1 = unspentReader1.readLine()) != null) {
                    String a1[] = line1.split("\t");
                    String add1 = a1[0];

                    String substring = add1.substring(0, binWidth);
                    if (!addBins.containsKey(substring))
                        addBins.put(substring, 1l);
                    else addBins.put(substring, 1l + addBins.get(substring));
                }
                unspentReader1.close();

            }
        }


        logger.info("Found " + addBins.size() + " address prefixes");
        logger.info(addBins.toString());
        for (String addprefix : addBins.keySet()) {
            aMap.clear();
            for (int year = 2009; year < 2018; year++) {
                for (int month = 1; month <= 12; month++) {
                    if (year == 2017 && month > 7) continue;
                    //if(year>2012)continue;

                    spentReader = new BufferedReader(new FileReader(dataDir + "inputs" + year + "_" + month + ".txt"));
                    unspentReader = new BufferedReader(new FileReader(dataDir + "unspent" + year + "_" + month + ".txt"));

                    String line = "";
                    while ((line = spentReader.readLine()) != null) {
                        String a[] = line.split("\t");
                        String add = a[0];
                        if (!addprefix.equals(add.substring(0, binWidth))) continue;
                        try {
                            BigInteger amount = new BigInteger(a[1]);
                        } catch (Exception e) {
                            logger.error(year + "_" + month + "_" + line);
                        }
                        BigInteger amount = new BigInteger(a[1]);
                        DateTime earnedTime = new DateTime(1000 * Long.parseLong(a[2]));
                        DateTime spentTime = new DateTime(1000 * Long.parseLong(a[3]));

                        int eMonth = earnedTime.monthOfYear().get();
                        int eYear = earnedTime.year().get();
                        int sYear = spentTime.year().get();
                        int sMonth = spentTime.monthOfYear().get();
                        if (eYear == sYear && eMonth == sMonth) {
                            //do nothing, spent already
                            useless++;
                        } else {
                            useful++;


                            int fromIndex = computeIndex(eMonth, eYear);
                            int toIndex = computeIndex(sMonth, sYear);
                            for (int i = fromIndex; i < toIndex; i++) {
                                addToAddress(aMap, add, fromIndex, toIndex, amount);
                            }
                        }


                    }

                    while ((line = unspentReader.readLine()) != null) {
                        String a[] = line.split("\t");
                        String add = a[0];
                        if (!addprefix.equals(add.substring(0, binWidth))) continue;
                        DateTime earnedTime = new DateTime(1000 * Long.parseLong(a[1]));
                        BigInteger amount = new BigInteger(a[2]);

                        int eMonth = earnedTime.monthOfYear().get();
                        int eYear = earnedTime.year().get();
                        int startIndex = computeIndex(eMonth, eYear);

                        addToAddress(aMap, add, startIndex, computeIndex(12, 2017), amount);
                    }
                    logger.info(addprefix + "\t" + year + "\t" + month + "\t" + useful + " " + useless + "\t" + addressCount);

                    int currentIndex = computeIndex(month, year);
                    HashMap<String, Integer> amountDist = new HashMap<>();
                    BufferedWriter wr = new BufferedWriter(new FileWriter(tempDir + addprefix + "_" + "dist" + year + "_" + month + ".txt"));
                    StringBuilder buf = new StringBuilder();
                    for (String address : new HashSet<>(aMap.keySet())) {
                        if (!aMap.get(address).containsKey(currentIndex)) {
                            //spent it all.
                            aMap.remove(address);
                        } else {
                            //add to the stats of this index
                            BigInteger balance = aMap.get(address).get(currentIndex);
                            String key = balance.toString();
                            if (!amountDist.containsKey(key)) {
                                amountDist.put(key, 0);
                            }
                            amountDist.put(key, 1 + amountDist.get(key));
                        }
                    }
                    for (String k : amountDist.keySet()) {
                        buf.append(k + "\t" + amountDist.get(k) + "\r\n");
                    }
                    wr.append(buf.toString());
                    wr.close();

                }
            }
        }


    }

    private static int computeIndex(int month, int year) {
        return (year - 2009) * 12 + (month);
    }

    private static void addToAddress(Map<String, HashMap<Integer, BigInteger>> aMap, String add, int fromIndex, int toIndex, BigInteger newCoins) {
        if (newCoins.equals(new BigInteger("0"))) return;
        if (!aMap.containsKey(add)) {
            aMap.put(add, new HashMap<Integer, BigInteger>());
        }

        for (int i = fromIndex; i < toIndex; i++) {
            if (!aMap.containsKey(i)) {
                aMap.get(add).put(i, newCoins);
            } else {
                BigInteger existingCoins = aMap.get(add).get(i);
                aMap.get(add).put(i, newCoins.add(existingCoins));
            }
        }
    }
}
