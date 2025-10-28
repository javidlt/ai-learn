#include <stdio.h>
#include <stdlib.h>

int main(){
    float f_val = 5.75f;
    int i_val = (int)f_val; // cast float to int
    printf("Float value: %.2f casted to Int value: %d\n", f_val, i_val);
    
    double d_val = 9.99;
    int i_from_d = (int)d_val; // cast double to int
    printf("Double value: %.2f casted to Int value: %d\n", d_val, i_from_d);
    
    char c_from_i = (char)i_val; // cast int to char
    printf("Int value: %d casted to Char value: %c\n", i_val, c_from_i);
    
    // Additional casting examples
    char c_val = 'A';
    int i_from_c = (int)c_val; // cast char to int (ASCII value)
    printf("Char value: %c casted to Int value: %d\n", c_val, i_from_c);
    
    int large_int = 300;
    char c_from_large = (char)large_int; // potential data loss
    printf("Large int: %d casted to Char: %d\n", large_int, c_from_large);
    
    return 0;
}