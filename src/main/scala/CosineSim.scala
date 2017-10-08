

object CosineSim {

  /*
   * This method takes 2 equal length arrays of integers
   * It returns a double representing similarity of the 2 arrays
   * 0.9925 would be 99.25% similar
   * (x dot y)/||X|| ||Y||
   */
  def cosineSimilarity(x: Array[Long], y: Array[Long]): Double = {
    require(x.size == y.size)
    val x0 = dotProduct(x, y)
    val x1 = math.sqrt(magn(x))
    val x2 = math.sqrt(magn(y))
    x0 / (x1 * x2)
  }

  /*
   * Return the dot product of the 2 arrays
   * e.g. (a[0]*b[0])+(a[1]*a[2])
   */
  def dotProduct(x: Array[Long], y: Array[Long]): Long = {
    (for ((a, b) <- x zip y) yield a * b) sum
  }

  /*
   * Return the magnitude of an array
   * We multiply each element, sum it, then square root the result.
   */
  def magn(x: Array[Long]): Double = {
    x.map(e => e * e).sum
  }

}