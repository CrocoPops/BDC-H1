import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import scala.Tuple3;
import java.io.*;
import java.util.*;


public class Homework1{

    public static void main(String[] args) throws IOException {
        // Check if the number of arguments is correct
        System.out.println(args.length);
        if (args.length != 5)
            throw new IllegalArgumentException("Wrong number of params!");

        // Read all points in the file and add them to the  (integers), and must compute and print the following information.list of points
        Scanner scanner = new Scanner(new File(args[0]));
        List<Point> points = new ArrayList<>();
        while (scanner.hasNextLine()) {
            String[] cords = scanner.nextLine().split(",");
            points.add(new Point(Float.parseFloat(cords[0]), Float.parseFloat(cords[1])));
        }

        // Read D, M, K, L parameters
        float D = Float.parseFloat(args[1]);
        int M = 1; // Default
        int K = 1; // Default
        int L = 1; //default

        try {
            M = Integer.parseInt(args[2]);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Param M is not an integer!");
        }

        try {
            K = Integer.parseInt(args[3]);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Param K is not an integer!");
        }

        ExactOutliers(points, D, M, K);

        try {
            L = Integer.parseInt(args[4]);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Param L is not integer!");
        }

        SparkConf conf = new SparkConf(true).setAppName("OutlierDetector");
        JavaRDD<String> docs;
        try (JavaSparkContext sc = new JavaSparkContext(conf)) {
            sc.setLogLevel("WARN");
            // Divide the inputFile in L partitions (each line is assigned to a specific partition
            docs = sc.textFile(args[0]).repartition(L).cache();
        }
        MRApproxOutliers(docs, D, M, K);
    }

    /**
     * Algorithm 2
     * @param docs - the RDD containing the document with the coordinates
     * @param D - distance from the point
     * @param M - number of points
     * @param K - number of outliers to print
     */
    public static void MRApproxOutliers(JavaRDD<String> docs, float D, int M, int K) {
        // Step A: transforms the input RDD into a RDD whose elements corresponds to the non-empty cells and, contain,
        // for each cell, its identifier (i, j) and the number of points of S that it contains.
        // Computation in Spark partitions, without gathering together all points of a cell.

        // ROUND 1
        JavaPairRDD<Tuple2<Integer, Integer>, Long> cell = docs.flatMapToPair(document -> { // <-- MAP PHASE (R1)
            String[] coord = document.split(",");
            HashMap<Tuple2<Integer, Integer>, Long> counts = new HashMap<>();
            ArrayList<Tuple2<Tuple2<Integer, Integer>, Long>> pairs = new ArrayList<>();
            // Compute the cells coordinates
            double lambda = D/(2*Math.sqrt(2));
            Tuple2<Integer, Integer> key = new Tuple2<>((int) Math.floor(Float.parseFloat(coord[0])/lambda), (int) Math.floor(Float.parseFloat(coord[1])/lambda));
            counts.put(key, 1L);
            for(Map.Entry<Tuple2<Integer, Integer>, Long> e : counts.entrySet()) {
                Tuple2<Tuple2<Integer, Integer>, Long> tuple = new Tuple2<>(e.getKey(), e.getValue());
                pairs.add(tuple);
            }
            return pairs.iterator();
        })

        .groupByKey()// <--SHUFFLE + GROUPING
        .mapToPair(pair -> { // <-- REDUCE PHASE (R1)
            Tuple2<Integer, Integer> cellKey = pair._1();
            Iterable<Long> counts = pair._2();
            // Compute the total points in a specific cell in this partition
            long sum = 0L;
            for (long count : counts) {
                sum += count;
            }
            return new Tuple2<>(cellKey, sum);
        }).cache();

        // Step B: transforms the RDD of cells, resulting from Step A, by attaching to each element, relative to a
        // non-empty cell C, the values |N3(C)| and |N7(C)|, as additional info. To this purpose, you can assume that
        // the total number of non-empty cells is small with respect to the capacity of each executor's memory.

        // Round 2
        JavaPairRDD<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>> cellNeighbors = cell.mapToPair(pair -> {
            int i = pair._1()._1();
            int j = pair._1()._2();
            long totalCount = pair._2();
            long N3 = 0L;
            long N7 = 0L;


            for(int dx = -3; dx <= 3; dx++) {
                for(int dy = -3; dy <= 3; dy++) {
                    Tuple2<Integer, Integer> neighborKey = new Tuple2<>(i + dx, j + dy);
                    long neighborCount = 0L;

                    // Search the neighborKey in the RDD
                    List<Long> neighborCountList = cell.lookup(neighborKey);
                    if (!neighborCountList.isEmpty()) {
                        Iterator<Long> iter =neighborCountList.iterator();

                        if (iter.hasNext()) {
                            neighborCount = iter.next();
                        }

                        if ((Math.abs(dx) <= 1) && (Math.abs(dy) <= 1)) {
                            N3 += neighborCount;
                        }
                        N7 += neighborCount;
                    } else System.out.println("Empty");
                }

            }

            Tuple3<Long, Long, Long> counts = new Tuple3<>(totalCount, N3, N7);

            return new Tuple2<>(new Tuple2<>(i, j), counts);
        }).cache();

        // Print:
        // - Number of sure (D, M) - outliers
        // - Number of uncertain points
        // - First K non-empty cells, in non-decreasing order of |N3(C)|, their identifiers and value of |N3(C)|, one
        //   line per cell.
        long sureOutliers = cellNeighbors.filter(triple -> triple._2()._3() <= M).count();
        long uncertainOutliers = cellNeighbors.filter(triple -> triple._2()._2() <= M && triple._2()._3() > M).count();

        System.out.println("Number of sure (" + D + "," + M + ") - outliers: " + sureOutliers);
        System.out.println("Number of uncertain points: " + uncertainOutliers);

        List<Tuple2<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>>> topKCells =
                cellNeighbors.takeOrdered(K, (t1, t2) -> Long.compare(t1._2()._2(), t2._2()._2()));

        System.out.println("First " + K + " non-empty cells in non-decreasing order of |N3(C)|: ");

        for(Tuple2<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>> cellInfo : topKCells)
            System.out.println("Cell: (" + cellInfo._1()._1() + ", " + cellInfo._1()._2() + "). |N3(C)|: " + cellInfo._2()._2());
    }

    /**
     * Algorithm 1
     * @param points -
     * @param D
     * @param M
     * @param K
     */
    public static void ExactOutliers(List<Point> points, float D, int M, int K) {
        ArrayList<Point> outliers = new ArrayList<>();

        // Calculate the distances
        // for a total of N(N-1)/2 distances being calculated
        double[][] distances = new double[points.size()][points.size()];
        for (int i = 0; i < points.size(); i++) {
            for (int j = i + 1; j < points.size(); j++) {
                distances[i][j] = points.get(i).distanceTo(points.get(j));
                distances[j][i] = distances[i][j];
            }
        }

        // Counting close (<= D) points for each point and if they are < M then
        // the point gets saved into a list as outlier
        for (int i = 0; i < points.size(); i++) {
            int count = 0;
            for (int j = 0; j < points.size(); j++)
                if (distances[i][j] <= D)
                    count++;

            if (count < M) {
                outliers.add(points.get(i));
            }
        }

        // Sort outliers based on number of closest points
        outliers.sort((p1, p2) -> {
            int count1 = 0, count2 = 0;
            int index1 = points.indexOf(p1);
            int index2 = points.indexOf(p2);
            for (int j = 0; j < points.size(); j++) {
                if (distances[index1][j] <= D)
                    count1++;

                if (distances[index2][j] <= D)
                    count2++;

            }
            return Integer.compare(count1, count2);
        });

        // Print the number of (D,M)-outliers
        System.out.println("Number of (D, M)-outliers: " + outliers.size());

        // Print the first K outliers
        System.out.println("First " + K + " outliers:");
        for (int i = 0; i < Math.min(K, outliers.size()); i++)
            System.out.println("(" + outliers.get(i).getX() + ", " + outliers.get(i).getY() + ")");
    }
}