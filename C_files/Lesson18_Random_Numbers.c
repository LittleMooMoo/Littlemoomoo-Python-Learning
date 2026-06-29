// Lesson18 Random Numbers
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    srand(time(NULL));
    int min = 100;
    int max = 900;

    int randomNum = (rand() % max + min);
    printf("%d", randomNum);

    return 0;
}
