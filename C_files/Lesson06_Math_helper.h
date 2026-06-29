// Lesson06 helper functions
double calculated_error(double target, double measurement)
{
    double error = target - measurement;
    return error;
}

double calculated_error_step(double initial_error, double new_error)
{
    double error_step = new_error - initial_error;
    return error_step;
}

double calculated_damping(double error_step, double time_step, float kd)
{
    double damping = kd * error_step / time_step;
    return damping;
}

double calculated_drive_force(double error, float kp)
{
    double spring_force = kp * error;
    return spring_force;
}
