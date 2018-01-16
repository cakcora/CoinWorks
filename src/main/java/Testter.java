import java.util.Arrays;

/**
 * Created by cxa123230 on 1/12/2018.
 */
public class Testter {

    public static void main(String[] args) throws Exception {
        Integer[][] occM = new Integer[5][5];
        Long[][] amoM = new Long[5][5];
        for (Integer[] row : occM)
            Arrays.fill(row, 0);
        for (Long[] row : amoM)
            Arrays.fill(row, 0L);
        System.out.println(occM[3][2]);
        System.out.println(amoM[3][2]);
    }


}
