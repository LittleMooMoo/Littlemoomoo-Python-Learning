// Lesson025_struct
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

/*typedef struct
{
    double x;
    double y;
    double z;
    char trajectory_name[50];
    float height;
    bool isPlanar;
} motion;

int main()
{

    // struct = A custom containner that holds multiple pieces of related information;
    //  struct is like a class / objects in python;
    motion motion1 = {5.002, 4.2323, 4.220, "cycloidal", true};
    motion motion2 = {3.2, 412.222, 2301.33, "straight jump", true};

    printf("%lf \n", motion1.x);
    return 0;
}*/

// arrays of struct is an array where each element is an array.

typedef struct
{
    char horn[25];
    int age;
    float weight;
} Cow;
int main()
{
    Cow cow_moo_list[] = {{"big horn", 2, 50},
                          {"small horn", 1, 25},
                          {"no horn", 4, 100}};

    /*Cow moo1 = {"big horn", 2, 50};
    Cow moo2 = {"small horn", 1, 25};
    Cow moo3 = {"no horn", 4, 100};*/
    int step = sizeof(Cow) / sizeof(cow_moo_list[0]);
    for (int i = 0; i < 3; i++)
    {
        printf("%s %d %.1f \n", cow_moo_list[i].horn, cow_moo_list[i].age, cow_moo_list[i].weight);
    }
    return 0;
}