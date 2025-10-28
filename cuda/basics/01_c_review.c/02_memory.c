// Practical example using reallocate to create a dynamic array
#include <stdio.h>
#include <stdlib.h>

struct list {
    int qty;
    int size;
    int *data;
};

void addNumber(struct list *l, int number);

int main() {
    struct list myList;
    myList.qty = 0;
    myList.size = 10;
    myList.data = malloc(sizeof(int) * myList.size);

    addNumber(&myList, 5);
    addNumber(&myList, 10);
    addNumber(&myList, 15);

    for (int i = 0; i < myList.qty; i++) {
        printf("%d\n", myList.data[i]);
    }

    free(myList.data);
    return 0;
}

void addNumber(struct list *l, int number) {
    l->qty += 1;
    if (l->qty > l->size) {
        l->size *= 2;
        l->data = realloc(l->data, sizeof(int) * l->size);
    }
    l->data[l->qty - 1] = number;
}