#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159
#define AREA(r) (PI * r * r)

#ifndef radius
#define radius 7
#endif

// if elif else logic
// we can only use integer constants in #if and #elif
#if radius > 10
#define radius 10
#elif radius < 5
#define radius 5
#else
#define radius 7
#endif

int main() {
    float area = AREA(radius);
    printf("Radius: %d\n", radius);
    printf("Area of circle with radius %d: %.2f\n", radius, area);
    return 0;
}
