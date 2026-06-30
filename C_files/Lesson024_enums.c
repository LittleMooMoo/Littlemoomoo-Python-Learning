// Lesson024_enums
#include <stdio.h>

// enum = A user-defined data typ ethat consists of a set of name integer constants.
// Benefits: Replaces numbers with readable names;

/*enum Day
{
    Sunday,
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday
};

int main()
{
    enum Day today = Sunday;

    char *mooday[30] = {
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday"};
    printf("%s", mooday[0]);
    printf("%d", today);
    return 0;
}*/
typedef enum
{
    GOO,
    GA,
    DING
} GUA;
void mooGUA(GUA gua);

int main()
{
    GUA gua = GOO;
    mooGUA(gua);

    return 0;
}

void mooGUA(GUA gua)
{
    printf("%d", gua + 1);
}