import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class MatrixJustification {
    private static final Logger logger = LoggerFactory.getLogger(MatrixJustification.class);

    public static void main(String[] args) throws IOException {

        String dir = args[0];
        String[] coins = {("Bitcoin"), ("Litecoin"), ("Namecoin")};
        BufferedReader br;
        for (String coin : coins) {
            double totalTx = 0;
            long matrix[][] = new long[100][100];
            int iMax = 0, oMax = 0;
            for (int year = 2009; year < 2018; year++) {
                for (int day = 1; day < 366; day++) {
                    String fileName = dir + coin + "/createddata/daily/" + year + "_" + day + ".txt";
                    File f = new File(fileName);
                    if (!f.exists()) {
//                       System.out.println(fileName + " does not exist");
                        continue;
                    }
                    br = new BufferedReader(new FileReader(fileName));
                    String line = "";
                    while ((line = br.readLine()) != null) {
                        totalTx++;
                        String[] split = line.split("\t");
                        int input = Integer.parseInt(split[2]);
                        int output = Integer.parseInt(split[3]);
                        if (input > iMax) {
                            iMax = input;
                        }
                        if (output > oMax) {
                            oMax = output;
                        }
                        if (input > matrix.length) input = matrix.length;
                        if (output > matrix.length) output = matrix.length;
                        matrix[input - 1][output - 1]++;
                    }
                }
            }
            //logger.info(coin+" "+iMax+" "+oMax);
            for (int i = 2; i < 50; i++) {
                int subTotal = 0;
                for (int j = 0; j < i; j++) {
                    for (int k = 0; k < i; k++) {
                        subTotal += matrix[j][k];
                    }
                }

                //System.out.println(coin+"\t"+i+"\t"+subTotal/(double)totalTx);
            }
            System.out.println(coin + " " + totalTx);
            for (int i = 0; i <= 20; i++) {
                for (int j = 0; j <= 20; j++) {
                    System.out.print(100 * matrix[i][j] / totalTx + "\t");
                }
                System.out.println();
            }
        }


    }
}
