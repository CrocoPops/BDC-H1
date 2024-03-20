import java.io.*;
import java.util.*;

public class Homework1{

    static List<Point> points = new ArrayList<Point>();

    public static void main(String[] args) throws IOException {

        // Check if the number of arguments is correct
        System.out.println(args.length);
        if(args.length != 1)
            throw new IllegalArgumentException("Wrong number of params!");

        // Read all points in the file and add them to the list of points
        Scanner scanner = new Scanner(new File(args[0]));
        while (scanner.hasNextLine()) {
            String[] cords = scanner.nextLine().split(",");
            points.add(new Point(Float.parseFloat(cords[0]), Float.parseFloat(cords[1])));
            System.out.println(cords[0] + ", " + cords[1]);
        }

    }

    public static void ExactOutliers(float D, int M, int K) {

    }

}