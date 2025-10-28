#include <stdio.h>
#include <stdlib.h>

// define struct 
struct Point {
    float x;
    float y;
};

// define typedef
typedef struct Point Point;

int main() {
    // create and initialize a Point instance
    Point p1;
    p1.x = 3.5f;
    p1.y = 4.5f;

    // print the coordinates
    printf("Point coordinates: (%.2f, %.2f)\n", p1.x, p1.y);

    return 0;
}