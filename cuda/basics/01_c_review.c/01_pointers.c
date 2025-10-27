#include <stdio.h>
#include <stdlib.h>

int main() {
    int a = 10;
    int b = 20;
    int *p1 = &a; // Pointer to a
    int *p2 = &b; // Pointer to b

    printf("Before swapping:\n");
    printf("a = %d, b = %d\n", a, b);

    // Swapping values using pointers
    int temp = *p1;
    *p1 = *p2;
    *p2 = temp;

    printf("After swapping:\n");
    printf("a = %d, b = %d\n", a, b);

    return 0;
}