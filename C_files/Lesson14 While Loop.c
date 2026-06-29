// Lesson14 While Loop
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

int main()
{
    /*int step = 0;

    while (step <= 0)
    {
        printf("enter a number greater than 0: ");
        scanf("%d", &step);
    }
    return 0;*/

    /*char name[50] = "";
    printf("enter your name:");
    fgets(name, sizeof(name), stdin);
    name[strlen(name) - 1] = '\0';
    while (strlen(name) == 0)
    {
        printf("Name cannot be empty, please enter your name: ");
        fgets(name, sizeof(name), stdin);
        name[strlen(name) - 1] = '\0';
    }*/
    bool is_online = true;
    char ping_response = '\0';

    while (is_online)
    {
        printf("you are online moo\n");
        printf("would you like to keep staying online moo? Y for yes, N for No : \n");
        scanf(" %c", &ping_response);
        if (ping_response == 'N' || ping_response == 'n')
        {
            is_online = false;
            printf("bye bye moo moo see you offline! \n");
        }
        else if (ping_response == 'Y' || ping_response == 'y')
        {
            printf("moo moo continue to be online\n");
            is_online = true;
        }
        else
        {
            printf("invalid input, program stopped.");
            is_online = false;
        }
    }
    return 0;
}
