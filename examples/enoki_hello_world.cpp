#include <fenv.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>

#include "opengl_window/tiny_opengl3_app.h"

#define USE_MATPLOTLIB 1

#ifdef USE_MATPLOTLIB
#include "third_party/matplotlib-cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

#include "../src/multi_body.hpp"
#include "../src/urdf/tiny_system_constructor.h"

#ifdef USE_MATPLOTLIB
template <typename Algebra>
void plot_trajectory(const std::vector<typename Algebra::VectorX> &states,
                     const std::string &title = "Figure") {
  for (int i = 0; i < static_cast<int>(states[0].size()); ++i) {
    std::vector<double> traj(states.size());
    for (int t = 0; t < static_cast<int>(states.size()); ++t) {
      traj[t] = Algebra::to_double(states[t][i]);
    }
    plt::named_plot("state[" + std::to_string(i) + "]", traj);
  }
  plt::legend();
  plt::title(title);
  plt::show();
}
#endif

template <typename Algebra>
void visualize_trajectory(const std::vector<typename Algebra::VectorX> &states,
                          MultiBody<Algebra> &mb, double dt,
                          const std::string &window_title = "Trajectory") {
  typedef ::Transform<Algebra> Transform;
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;

  TinyOpenGL3App app(window_title.c_str(), 1024, 768);
  app.m_renderer->init();
  app.set_up_axis(2);
  app.m_renderer->get_active_camera()->set_camera_distance(4);
  app.m_renderer->get_active_camera()->set_camera_pitch(-30);
  app.m_renderer->get_active_camera()->set_camera_target_position(0, 0, 0);

  for (std::size_t i = 0; i < mb.links.size(); ++i) {
    int cube_shape = app.register_cube_shape(0.1f, 0.1f, 0.1f);
    int cube_id = app.m_renderer->register_graphics_instance(cube_shape);
    mb.links[i].visual_ids = {cube_id};
    mb.links[i].X_visuals = {Transform(mb.links[i].rbi.com)};
  }

  for (const std::vector<double> &state : states) {
    app.m_renderer->update_camera(2);
    DrawGridData data;
    data.upAxis = 2;
    app.draw_grid(data);
    for (int i = 0; i < mb.dof(); ++i) {
      mb.q[i] = state[i];
    }
    mb.forward_kinematics();
    // mb.print_state();

    std::this_thread::sleep_for(std::chrono::duration<double>(dt));

    TinyVector3f parent_pos(static_cast<float>(mb.base_X_world.translation[0]),
                            static_cast<float>(mb.base_X_world.translation[1]),
                            static_cast<float>(mb.base_X_world.translation[2]));
    for (const auto &link : mb.links) {
      TinyVector3f link_pos(static_cast<float>(link.X_world.translation[0]),
                            static_cast<float>(link.X_world.translation[1]),
                            static_cast<float>(link.X_world.translation[2]));

      app.m_renderer->draw_line(link_pos, parent_pos,
                                TinyVector3f(0.5, 0.5, 0.5), 2.f);
      parent_pos = link_pos;
      for (std::size_t j = 0; j < link.visual_ids.size(); ++j) {
        Transform X_visual = link.X_world * link.X_visuals[j];
        // sync transform
        TinyVector3f geom_pos(static_cast<float>(X_visual.translation[0]),
                              static_cast<float>(X_visual.translation[1]),
                              static_cast<float>(X_visual.translation[2]));
        auto quat = Algebra::matrix_to_quat(X_visual.rotation);
        TinyQuaternionf geom_orn(static_cast<float>(Algebra::quat_x(quat)),
                                 static_cast<float>(Algebra::quat_y(quat)),
                                 static_cast<float>(Algebra::quat_z(quat)),
                                 static_cast<float>(Algebra::quat_w(quat)));
        app.m_renderer->write_single_instance_transform_to_cpu(
            geom_pos, geom_orn, link.visual_ids[j]);
        TinyVector3f color(0.1, 0.6, 0.8);
        app.m_renderer->draw_line(link_pos, geom_pos, color, 2.f);
      }
    }
    app.m_renderer->render_scene();
    app.m_renderer->write_transforms();
    app.swap_buffer();
  }
}

int main(int argc, char **argv) {
  {
    using Tf = Transform<EnokiAlgebra>;
    using Vector3 = EnokiAlgebra::Vector3;
    using Matrix3 = EnokiAlgebra::Matrix3;
    using RigidBodyInertia = ::RigidBodyInertia<EnokiAlgebra>;

    // Set NaN trap
    // feenableexcept(FE_INVALID | FE_OVERFLOW);

    Tf tf;
    tf.set_identity();
    std::cout << "tf: " << tf << std::endl;

    EnokiAlgebra::Vector3 vec(1., 2, 3);
    std::cout << "VCM: " << EnokiAlgebra::cross_matrix(vec) << "\n";

    std::cout << "rot-x: " << EnokiAlgebra::rotation_x_matrix(0.3) << std::endl;

    RigidBodyInertia rbi;
    std::cout << "rbi.inertia: " << rbi.inertia << std::endl;

    EnokiAlgebra::Matrix6 mat6(0.);
    EnokiAlgebra::assign_block(mat6, EnokiAlgebra::Matrix3(3.14), 0, 2);
    std::cout << "mat6: " << mat6 << std::endl;

    double angle = M_PI_2;
    std::cout << "rotx(0):\n"
              << EnokiAlgebra::rotation_x_matrix(angle) << std::endl;

    // ArticulatedBodyInertia<EnokiAlgebra> abi;
    // abi.I = Matrix3(1.);
    // abi.H = Matrix3(2.);
    // abi.M = Matrix3(3.);
    // std::cout << "abi:\n" << abi.matrix() << std::endl;
    // ArticulatedBodyInertia<EnokiAlgebra> abi_sum = abi + abi.matrix();
    // std::cout << "abi+abi:\n" << abi_sum.matrix() << std::endl;
    // ArticulatedBodyInertia<EnokiAlgebra> abi_sub = abi - abi.matrix();
    // std::cout << "abi-abi:\n" << abi_sub.matrix() << std::endl;

    //   return 0;

    MultiBody<EnokiAlgebra> mb;

    double mass = 1.;
    Vector3 com(0., 0., 1.);
    Matrix3 I = EnokiAlgebra::diagonal3(Vector3(1., 1., 1.));
    Link<EnokiAlgebra> link_a(JOINT_REVOLUTE_Y, Tf(0., 0., 1.),
                              RigidBodyInertia(mass, com, I));
    Link<EnokiAlgebra> link_b(JOINT_REVOLUTE_Y, Tf(0., 0., 1.),
                              RigidBodyInertia(mass, com, I));
    mb.attach(link_a);
    mb.attach(link_b);
    mb.initialize();

    mb.q = {M_PI_2, 0.0};

    mb.forward_kinematics();
    Vector3 gravity(0., 0., -9.81);
    mb.forward_dynamics(gravity);

    std::vector<typename EnokiAlgebra::VectorX> traj;

    double dt = 0.01;
    for (int i = 0; i < 10000; ++i) {
      traj.push_back(mb.q);
      mb.integrate(dt);
      mb.forward_dynamics(gravity);
      mb.print_state();
    }

    // plot_trajectory<EnokiAlgebra>(traj);
    visualize_trajectory<EnokiAlgebra>(traj, mb, dt);
  }

  return 0;

  using namespace enoki;

  using FloatC = CUDAArray<float>;
  using FloatD = DiffArray<FloatC>;

  {
    FloatD a = 1.f;
    set_requires_gradient(a);

    FloatD b = erf(a);
    set_label(a, "a");
    set_label(b, "b");

    // std::cout << graphviz(b) << std::endl;

    backward(b);
    std::cout << gradient(a) << std::endl;
  }

  {
    /* Declare underlying packet type, could just be 'float' for scalar
     * arithmetic
     */
    using FloatP = float;  // Packet<float, 4>;

    /* Define vectorized quaternion type */
    using QuaternionP = Quaternion<FloatP>;

    QuaternionP a = QuaternionP(1.f, 0.f, 0.f, 0.f);
    QuaternionP b = QuaternionP(0.f, 1.f, 0.f, 0.f);

    /* Compute several rotations that interpolate between 'a' and 'b' */
    FloatP t = linspace<FloatP>(0.f, 1.f);
    std::cout << "t:  " << t << std::endl;
    QuaternionP c = slerp(a, b, t);

    std::cout << "Interpolated quaternions:" << std::endl;
    std::cout << c << std::endl << std::endl;

    /* Turn into a 4x4 homogeneous coordinate rotation matrix packet */
    using Matrix4P = Matrix<FloatP, 4>;
    Matrix4P c_rot = quat_to_matrix<Matrix4P>(c);

    std::cout << "Rotation matrices:" << std::endl;
    std::cout << c_rot << std::endl << std::endl;

    /* Round trip: turn the rotation matrices back into rotation quaternions */
    QuaternionP c2 = matrix_to_quat(c_rot);

    if (hsum(abs(c - c2)) < 1e-6f)
      std::cout << "Test passed." << std::endl;
    else
      std::cout << "Test failed." << std::endl;
  }
}