// Lesson05 User_Input
#include <stdio.h>
#include <string.h>

int main()
{
    int fixed_motor_id = 1;
    int input_motor_id;
    float motor_target;
    float measured_position = 5.2;
    float position_error;
    char motor_name[30] = "front right shank";
    char motor_name_input[30];
    // char moo = '\0';
    /*
    printf("%d\n", motor_id);
    printf("%f\n", motor_target);
    printf("%f\n", motor_target);
    printf("%s\n", motor_name);
    */
    // printf("%c\n", moo);
    printf("please enter the motor position target: ");
    scanf("%f", &motor_target);

    printf("please enter the motor id : ");
    scanf("%d", &input_motor_id);

    position_error = motor_target - measured_position;
    printf("position error: %.2f\n", position_error);

    getchar();
    printf("please enter the motor name: ");
    fgets(motor_name_input, sizeof(motor_name_input), stdin);
    motor_name_input[strcspn(motor_name_input, "\n")] = '\0';

    if (strcmp(motor_name_input, motor_name) == 0)
    {
        printf("motor found\n");
        printf("corresponding motor id is: %d \n", fixed_motor_id);
    }
    else
    {
        printf("motor id not found\n");
    }

    if (input_motor_id == 1)
    {
        printf(" motor is found, ");
        printf("corresponding motor name is : '%s' \n", motor_name);
    }
    else
    {
        printf("motor is not found");
    }

    return 0;
}
