package Anomaly;

import java.util.Objects;

public class Output {
    String txID;
    String address;
    int index;
    long amount;

    public Output(String tx, String addr, int ind, long amo) {
        txID = tx;
        address = addr;
        index = ind;
        amount = amo;
    }

    public Output(String prevTx, int index) {
        this.txID = prevTx;
        this.index = index;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Output output = (Output) o;
        return index == output.index &&
                Objects.equals(txID, output.txID);
    }

    @Override
    public int hashCode() {

        return Objects.hash(txID, index);
    }

    @Override
    public String toString() {
        return txID + ' ' + index;
    }

    public long getAmount() {
        return amount;
    }
}
