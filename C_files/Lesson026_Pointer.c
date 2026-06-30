// Lesson026_Pointer
#include <stdio.h>
int moo;

void mooday(int *moo);

int main()
{
    // pointer = A variable that stores the memory address of another variable
    // Benefit : help avoid wasting memory by allowing to pass the adress
    //           of a large data strcuture instead of copying the entire data.

    int moo = 23;
    int *pmoo = &moo;
    // printf("%p \n", &moo);
    // printf("%p \n", pmoo);
    mooday(pmoo);

    printf("%d", moo);
    return 0;
}

void mooday(int *moo)
{ // pass by value
    // pass by reference
    (*moo)++;
}