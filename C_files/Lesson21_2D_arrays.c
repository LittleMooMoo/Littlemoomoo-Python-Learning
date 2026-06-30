// Lesson21_2D_arrays
#include <stdio.h>
#include <string.h>

/*int main()
{
    char numnums[][3] = {{'1', '2', '3'},
                         {'4', '5', '6'},
                         {'m', '&', '$'}};

    for (int i = 0; i < 3; i++) // rows
    {
        for (int j = 0; j < 3; j++) // columns
        {
            printf("  %c  ", numnums[i][j]); // print columns with two space front and back;
        }
        printf("\n"); // print rows with one row space in between;
    }

    return 0;
}

// numbers[row][column], row and column index all start from 0
*/

/*int main()
{
    char moomoograss[][10] = {"grass",
                              "Banana",
                              "gogogaga"};
    int size = sizeof(moomoograss) / sizeof(moomoograss[0]);
    moomoograss[0][0] = 'e';
    moomoograss[0][4] = 'A';

    for (int j = 0; j < size; j++) // columns
    {
        printf("  %s \n", moomoograss[j]); // print columns with two space front and back;
    }
    return 0;
}*/

int main()
{
    // exercise
    char names[4][25] = {0};
    int row_space = sizeof(names) / sizeof(names[0]);

    for (int j = 0; j < row_space; j++)
    {
        printf("please enter a name: ");
        fgets(names[j], sizeof(names[j]), stdin);
        names[j][strlen(names[j]) - 1] = '\0';
    }
    for (int i = 0; i < row_space; i++)
    {
        printf(" %s \n", names[i]);
    }

    return 0;
}
