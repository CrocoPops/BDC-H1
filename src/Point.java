public class Point {
    private final double x;
    private final double y;

    public Point(double x, double y){
        this.x = x;
        this.y = y;
    }

    public double getX(){
        return this.x;
    }

    public double getY(){
        return this.y;
    }

    public double distanceTo(Point other) {
        double deltaX = other.getX() - this.getX();
        double deltaY = other.getY() - this.getY();
        // return Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        return Math.sqrt(Math.pow(deltaX, 2) + Math.pow(deltaY, 2));
    }

}
