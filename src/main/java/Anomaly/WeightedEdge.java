package Anomaly;

import java.util.Objects;

public class WeightedEdge {

    long amo;
    int id;

    public WeightedEdge(long amo, int edgeNumber) {
        this.amo = amo;
        this.id = edgeNumber;

    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        WeightedEdge that = (WeightedEdge) o;
        return id == that.id;
    }

    @Override
    public int hashCode() {

        return Objects.hash(id);
    }

    public int getId() {
        return id;
    }

    public long getAmount() {
        return amo;
    }
}
