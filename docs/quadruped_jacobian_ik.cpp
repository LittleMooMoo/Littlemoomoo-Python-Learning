#include <Eigen/Dense>
#include <iostream>
#include <cmath>

struct leg_param
{
    const double shoulder_length;
    const double thigh_length;
    const double shank_length;
};
struct leg_joints
{
    double hip_joint_angle;
    double thigh_joint_angle;
    double shank_joint_angle;
};
struct foot_position
{
    double foot_x;
    double foot_y;
    double foot_z;
};
struct Jacobian_comp
{
    double dx_dhip;
    double dx_dthigh;
    double dx_dshank;
    double dy_dhip;
    double dy_dthigh;
    double dy_dshank;
    double dz_dhip;
    double dz_dthigh;
    double dz_dshank;
};
foot_position get_foot_position(const leg_param &leg, leg_joints &joints)
{
    double a1 = joints.thigh_joint_angle - joints.shank_joint_angle;
    double Z_1 = leg.thigh_length * std::cos(joints.thigh_joint_angle) + leg.shank_length * std::cos(a1);

    double x = leg.thigh_length * std::sin(joints.thigh_joint_angle) + leg.shank_length * std::sin(a1);
    double y = leg.shoulder_length * std::cos(joints.hip_joint_angle) + Z_1 * std::sin(joints.hip_joint_angle);
    double z = -leg.shoulder_length * std::sin(joints.hip_joint_angle) + Z_1 * std::cos(joints.hip_joint_angle);

    return {x, y, z};
};

Jacobian_comp calculate_jacobian_comp(const leg_param &leg, leg_joints &joints)
{
    double a1 = joints.thigh_joint_angle - joints.shank_joint_angle;
    double Z_1 = leg.thigh_length * std::cos(joints.thigh_joint_angle) + leg.shank_length * std::cos(a1);
    double Z_2 = -leg.thigh_length * std::sin(joints.thigh_joint_angle) - leg.shank_length * std::sin(a1);

    double dx_dh = 0;
    double dx_dt = leg.thigh_length * std::cos(joints.thigh_joint_angle) + leg.shank_length * std::cos(a1);
    double dx_ds = -leg.shank_length * std::cos(a1);

    double dy_dh = -leg.shoulder_length * std::sin(joints.hip_joint_angle) + std::cos(joints.hip_joint_angle) * Z_1;
    double dy_dt = std::sin(joints.hip_joint_angle) * Z_2;
    double dy_ds = leg.shank_length * std::sin(joints.hip_joint_angle) * std::sin(a1);

    double dz_dh = -leg.shoulder_length * std::cos(joints.hip_joint_angle) - std::sin(joints.hip_joint_angle) * Z_1;
    double dz_dt = std::cos(joints.hip_joint_angle) * Z_2;
    double dz_ds = leg.shank_length * std::cos(joints.hip_joint_angle) * std::sin(a1);

    return {dx_dh, dx_dt, dx_ds, dy_dh, dy_dt, dy_ds, dz_dh, dz_dt, dz_ds};
}

Eigen::Matrix3d get_forward_jacobian(Jacobian_comp &J_comp)
{
    Eigen::Matrix3d Forward_J;
    Forward_J << J_comp.dx_dhip, J_comp.dx_dthigh, J_comp.dx_dshank,
        J_comp.dy_dhip, J_comp.dy_dthigh, J_comp.dy_dshank,
        J_comp.dz_dhip, J_comp.dz_dthigh, J_comp.dz_dshank;
    return Forward_J;
};
Eigen::Vector3d get_foot_pos_vector(foot_position &foot_pose)
{

    Eigen::Vector3d foot_pose_vector;
    foot_pose_vector << foot_pose.foot_x, foot_pose.foot_y, foot_pose.foot_z;
    return foot_pose_vector;
};

int main()
{
    const leg_param leg = {84.26, 270.0, 270.0};
    leg_joints input_joint_pose;
    foot_position foot_pose;
    std::cout << "please enter a joint posture in hip, thigh, shank : " << std::endl;
    std::cin >> input_joint_pose.hip_joint_angle >> input_joint_pose.thigh_joint_angle >> input_joint_pose.shank_joint_angle;
    Jacobian_comp J_comp = calculate_jacobian_comp(leg, input_joint_pose);
    foot_pose = get_foot_position(leg, input_joint_pose);

    Eigen::Vector3d foot_pose_vector = get_foot_pos_vector(foot_pose);
    Eigen::Matrix3d Forward_J = get_forward_jacobian(J_comp);

    std::cout << "this is the jacobian at this given set of parameters: \n"
              << Forward_J << std::endl;
    std::cout << "this is the foot position vector at the given joint input: \n"
              << foot_pose_vector;
    return 0;
}