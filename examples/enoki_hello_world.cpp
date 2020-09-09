#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>

// #define DEBUG 1

#include "opengl_window/tiny_opengl3_app.h"

// #define USE_MATPLOTLIB 1

#ifdef USE_MATPLOTLIB
#include "third_party/matplotlib-cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

#include "base.hpp"
#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "dynamics/jacobian.hpp"
#include "dynamics/kinematics.hpp"
#include "math/tiny/tiny_algebra.hpp"
#include "math/tiny/tiny_double_utils.h"
#include "multi_body.hpp"
#include "urdf/urdf_cache.hpp"
#include "utils/file_utils.hpp"
#include "world.hpp"

#define USE_RBDL true

#if USE_RBDL
#include "rbdl/Dynamics.h"
#include "rbdl/Model.h"
#include "rbdl/rbdl.h"
#endif

using namespace tds;

const double ERROR_TOLERANCE = 1e-4;

#ifdef USE_MATPLOTLIB
template <typename Algebra>
void plot_trajectory(const std::vector<typename Algebra::VectorX> &states,
                     const std::string &title = "Figure") {
  for (int i = 0; i < Algebra::size(states[0]); ++i) {
    std::vector<double> traj(states.size());
    for (int t = 0; t < static_cast<int>(states.size()); ++t) {
      traj[t] = Algebra::to_double(states[t][i]);
    }
    plt::named_plot("state[" + std::to_string(i) + "]", traj);
  }
  plt::legend();
  plt::grid(true);
  plt::title(title);
  plt::show();
}
#endif

template <typename Algebra>
void visualize_trajectory(const std::vector<typename Algebra::VectorX> &states,
                          MultiBody<Algebra> &mb, double dt,
                          std::size_t skip_steps = 10,
                          const std::string &window_title = "Trajectory") {
  typedef Transform<Algebra> Transform;
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using VectorX = typename Algebra::VectorX;
  using Matrix3 = typename Algebra::Matrix3;

  TinyOpenGL3App app(window_title.c_str(), 1024, 768);
  app.m_renderer->init();
  app.set_up_axis(2);
  app.m_renderer->get_active_camera()->set_camera_distance(4);
  app.m_renderer->get_active_camera()->set_camera_pitch(-30);
  app.m_renderer->get_active_camera()->set_camera_target_position(0, 0, 0);

  float box_size = 0.02f;

  for (std::size_t i = 0; i < mb.size(); ++i) {
    int cube_shape = app.register_cube_shape(box_size, box_size, box_size);
    int cube_id = app.m_renderer->register_graphics_instance(cube_shape);
    mb[i].visual_ids = {cube_id};
    mb[i].X_visuals = {Transform(mb[i].rbi.com)};
  }

  for (std::size_t t = 0; t < states.size(); t += skip_steps) {
    const auto &state = states[t];
    app.m_renderer->update_camera(2);
    DrawGridData data;
    data.upAxis = 2;
    app.draw_grid(data);
    for (int i = 0; i < mb.dof(); ++i) {
      mb.q()[i] = state[i];
    }
    forward_kinematics(mb);
    mb.print_state();

    std::this_thread::sleep_for(std::chrono::duration<double>(dt));

    TinyVector3f parent_pos(
        static_cast<float>(mb.base_X_world().translation[0]),
        static_cast<float>(mb.base_X_world().translation[1]),
        static_cast<float>(mb.base_X_world().translation[2]));
    for (const auto &link : mb) {
      TinyVector3f link_pos(static_cast<float>(link.X_world.translation[0]),
                            static_cast<float>(link.X_world.translation[1]),
                            static_cast<float>(link.X_world.translation[2]));

      app.m_renderer->draw_line(link_pos, parent_pos,
                                TinyVector3f(0.5, 0.5, 0.5), 2.f);
      parent_pos = link_pos;
      for (std::size_t j = 0; j < link.visual_ids.size(); ++j) {
        Transform X_visual = link.X_world * link.X_visuals[j];
        Algebra::print("link.X_world", link.X_world);
        Algebra::print("X_visual", X_visual);
        Algebra::print("link.X_visuals[j]", link.X_visuals[j]);
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

#if USE_RBDL
template <typename Algebra>
RigidBodyDynamics::Math::Vector3d to_rbdl(const typename Algebra::Vector3 &v) {
  return RigidBodyDynamics::Math::Vector3d(Algebra::to_double(v[0]),
                                           Algebra::to_double(v[1]),
                                           Algebra::to_double(v[2]));
}

template <typename Algebra>
RigidBodyDynamics::Math::Matrix3d to_rbdl(const typename Algebra::Matrix3 &m) {
  return RigidBodyDynamics::Math::Matrix3d(
      Algebra::to_double(m(0, 0)), Algebra::to_double(m(0, 1)),
      Algebra::to_double(m(0, 2)), Algebra::to_double(m(1, 0)),
      Algebra::to_double(m(1, 1)), Algebra::to_double(m(1, 2)),
      Algebra::to_double(m(2, 0)), Algebra::to_double(m(2, 1)),
      Algebra::to_double(m(2, 2)));
}

template <typename Algebra>
RigidBodyDynamics::Math::SpatialTransform to_rbdl(
    const Transform<Algebra> &tf) {
  return RigidBodyDynamics::Math::SpatialTransform(
      to_rbdl<Algebra>(tf.rotation), to_rbdl<Algebra>(tf.translation));
}

template <typename Algebra>
RigidBodyDynamics::Model to_rbdl(const MultiBody<Algebra> &mb) {
  RigidBodyDynamics::Model model;
  if (mb.is_floating()) {
    RigidBodyDynamics::Body body(Algebra::to_double(mb.base_rbi().mass),
                                 to_rbdl<Algebra>(mb.base_rbi().com),
                                 to_rbdl<Algebra>(mb.base_rbi().inertia));
    RigidBodyDynamics::Joint joint;
    joint = RigidBodyDynamics::Joint(RigidBodyDynamics::JointTypeFloatingBase);
    model.AddBody(0, to_rbdl<Algebra>(tds::Transform<Algebra>()), joint, body);
  }
  for (const Link<Algebra> &link : mb) {
    RigidBodyDynamics::Body body(Algebra::to_double(link.rbi.mass),
                                 to_rbdl<Algebra>(link.rbi.com),
                                 to_rbdl<Algebra>(link.rbi.inertia));
    RigidBodyDynamics::Joint joint;
    switch (link.joint_type) {
      case JOINT_REVOLUTE_X:
      case JOINT_REVOLUTE_Y:
      case JOINT_REVOLUTE_Z:
      case JOINT_REVOLUTE_AXIS:
        joint = RigidBodyDynamics::Joint(RigidBodyDynamics::JointTypeRevolute,
                                         to_rbdl<Algebra>(link.S.top));
        break;
      case JOINT_PRISMATIC_X:
      case JOINT_PRISMATIC_Y:
      case JOINT_PRISMATIC_Z:
      case JOINT_PRISMATIC_AXIS:
        joint = RigidBodyDynamics::Joint(RigidBodyDynamics::JointTypePrismatic,
                                         to_rbdl<Algebra>(link.S.bottom));
        break;
      case JOINT_FIXED:
        joint = RigidBodyDynamics::Joint(RigidBodyDynamics::JointTypeFixed);
        break;
      default:
        joint = RigidBodyDynamics::Joint(RigidBodyDynamics::JointTypeUndefined);
        break;
    }
    unsigned int parent_id = link.parent_index < 0 ? 0u : link.parent_index + 1;
    if (mb.is_floating()) {
      // TODO fix this
      parent_id += 2;
    }
    model.AddBody(parent_id, to_rbdl<Algebra>(link.X_T), joint, body);
  }
  return model;
}

template <typename Algebra>
bool is_equal(const SpatialVector<Algebra> &a,
              const RigidBodyDynamics::Math::SpatialVector &b) {
  for (int i = 0; i < 6; ++i) {
    if (std::abs(Algebra::to_double(a[i]) - b[i]) > ERROR_TOLERANCE) {
      std::cout << "a[" << i << "] = " << Algebra::to_double(a[i]);
      std::cout << "\tb[" << i << "] = " << b[i];
      std::cout << "\terror = " << std::abs(Algebra::to_double(a[i]) - b[i])
                << std::endl;
      return false;
    }
  }
  return true;
}

template <typename Algebra>
bool is_equal(const typename Algebra::Vector3 &a,
              const RigidBodyDynamics::Math::Vector3d &b) {
  for (int i = 0; i < 3; ++i) {
    if (std::abs(Algebra::to_double(a[i]) - b[i]) > ERROR_TOLERANCE) {
      std::cout << "a[" << i << "] = " << Algebra::to_double(a[i]);
      std::cout << "\tb[" << i << "] = " << b[i];
      std::cout << "\terror = " << std::abs(Algebra::to_double(a[i]) - b[i])
                << std::endl;
      return false;
    }
  }
  return true;
}

template <typename Algebra>
bool is_equal(const typename Algebra::VectorX &a,
              const RigidBodyDynamics::Math::VectorNd &b) {
  for (typename Algebra::Index i = 0; i < Algebra::size(a); ++i) {
    if (std::abs(Algebra::to_double(a[i]) - b[i]) > ERROR_TOLERANCE) {
      std::cout << "a[" << i << "] = " << Algebra::to_double(a[i]);
      std::cout << "\tb[" << i << "] = " << b[i];
      std::cout << "\terror = " << std::abs(Algebra::to_double(a[i]) - b[i])
                << std::endl;
      return false;
    }
  }
  return true;
}

template <typename Algebra>
bool is_equal(const typename Algebra::Matrix3 &a,
              const RigidBodyDynamics::Math::Matrix3d &b) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (std::abs(Algebra::to_double(a(i, j)) - b(i, j)) > ERROR_TOLERANCE) {
        std::cout << "a[" << i << "," << j
                  << "] = " << Algebra::to_double(a(i, j));
        std::cout << "\tb[" << i << "," << j << "] = " << b(i, j);
        std::cout << "\terror = "
                  << std::abs(Algebra::to_double(a(i, j)) - b(i, j))
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

template <typename Algebra>
bool is_equal(const typename Algebra::MatrixX &a,
              const RigidBodyDynamics::Math::MatrixNd &b) {
  for (int i = 0; i < Algebra::num_rows(a); ++i) {
    for (int j = 0; j < Algebra::num_cols(a); ++j) {
      if (std::abs(Algebra::to_double(a(i, j)) - b(i, j)) > ERROR_TOLERANCE) {
        std::cout << "a[" << i << "," << j
                  << "] = " << Algebra::to_double(a(i, j));
        std::cout << "\tb[" << i << "," << j << "] = " << b(i, j);
        std::cout << "\terror = "
                  << std::abs(Algebra::to_double(a(i, j)) - b(i, j))
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

template <typename Algebra>
bool is_equal(const typename Algebra::Matrix3X &a,
              const RigidBodyDynamics::Math::MatrixNd &b) {
  for (int i = 0; i < Algebra::num_rows(a); ++i) {
    for (int j = 0; j < Algebra::num_cols(a); ++j) {
      if (std::abs(Algebra::to_double(a(i, j)) - b(i, j)) > ERROR_TOLERANCE) {
        std::cout << "a[" << i << "," << j
                  << "] = " << Algebra::to_double(a(i, j));
        std::cout << "\tb[" << i << "," << j << "] = " << b(i, j);
        std::cout << "\terror = "
                  << std::abs(Algebra::to_double(a(i, j)) - b(i, j))
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

template <typename Algebra>
bool is_equal(const Transform<Algebra> &a,
              const RigidBodyDynamics::Math::SpatialTransform &b) {
  return is_equal<Algebra>(a.translation, b.r) &&
         is_equal<Algebra>(a.rotation, b.E);
}

template <typename Algebra>
bool is_equal(const ArticulatedBodyInertia<Algebra> &a,
              const RigidBodyDynamics::Math::SpatialMatrix &b) {
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (std::abs(Algebra::to_double(a(i, j)) - b(i, j)) > ERROR_TOLERANCE) {
        std::cout << "a[" << i << "," << j
                  << "] = " << Algebra::to_double(a(i, j));
        std::cout << "\tb[" << i << "," << j << "] = " << b(i, j);
        std::cout << "\terror = "
                  << std::abs(Algebra::to_double(a(i, j)) - b(i, j))
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

template <typename Algebra>
bool is_equal(const MultiBody<Algebra> &tds,
              const RigidBodyDynamics::Model &rbdl) {
  // floating-base models in RBDL have a TranslationXYZ and a spherical joint
  // first
  std::size_t id_offset = tds.is_floating() ? 3 : 1;
  for (std::size_t j = 0; j < tds.size(); ++j) {
    std::size_t rbdl_j = j + id_offset;
    if (!is_equal<Algebra>(tds[j].S, rbdl.S[rbdl_j])) {
      fprintf(stderr, "Mismatch in S at link %i.\n", static_cast<int>(j));
      Algebra::print("TDS:  ", tds[j].S);
      std::cerr << "RBDL: " << rbdl.S[rbdl_j].transpose() << std::endl;
      return false;
    }
    if (!is_equal<Algebra>(tds[j].a, rbdl.a[rbdl_j])) {
      fprintf(stderr, "Mismatch in a at link %i.\n", static_cast<int>(j));
      Algebra::print("TDS:  ", tds[j].a);
      std::cerr << "RBDL: " << rbdl.a[rbdl_j].transpose() << std::endl;
      // return false;
    }
    if (!is_equal<Algebra>(tds[j].v, rbdl.v[rbdl_j])) {
      fprintf(stderr, "Mismatch in v at link %i.\n", static_cast<int>(j));
      Algebra::print("TDS:  ", tds[j].v);
      std::cerr << "RBDL: " << rbdl.v[rbdl_j].transpose() << std::endl;
      // return false;
    }
    if (!is_equal<Algebra>(tds[j].c, rbdl.c[rbdl_j])) {
      fprintf(stderr, "Mismatch in c at link %i.\n", static_cast<int>(j));
      Algebra::print("TDS:  ", tds[j].c);
      std::cerr << "RBDL: " << rbdl.c[rbdl_j].transpose() << std::endl;
      // return false;
    }
    if (!is_equal<Algebra>(tds[j].U, rbdl.U[rbdl_j])) {
      fprintf(stderr, "Mismatch in U at link %i.\n", static_cast<int>(j));
      Algebra::print("TDS:  ", tds[j].U);
      std::cerr << "RBDL: " << rbdl.U[rbdl_j].transpose() << std::endl;
      // return false;
    }
    if (!is_equal<Algebra>(tds[j].pA, rbdl.pA[rbdl_j])) {
      fprintf(stderr, "Mismatch in pA at link %i.\n", static_cast<int>(j));
      Algebra::print("TDS:  ", tds[j].pA);
      std::cerr << "RBDL: " << rbdl.pA[rbdl_j].transpose() << std::endl;
      // return false;
    }
    if (!is_equal<Algebra>(tds[j].abi, rbdl.IA[rbdl_j])) {
      fprintf(stderr, "Mismatch in ABI at link %i.\n", static_cast<int>(j));
      Algebra::print("TDS:  ", tds[j].abi);
      std::cerr << "RBDL:\n" << rbdl.IA[rbdl_j] << std::endl;
      return false;
    }
    if (!is_equal<Algebra>(tds[j].X_world, rbdl.X_base[rbdl_j])) {
      fprintf(stderr, "Mismatch in X_base (X_world) at link %i.\n",
              static_cast<int>(j));
      Algebra::print("TDS:  ", tds[j].X_world);
      std::cerr << "RBDL:\n" << rbdl.X_base[rbdl_j] << std::endl;
      // return false;
    }
    if (!is_equal<Algebra>(tds[j].X_parent, rbdl.X_lambda[rbdl_j])) {
      fprintf(stderr, "Mismatch in X_lambda (X_parent) at link %i.\n",
              static_cast<int>(j));
      Algebra::print("TDS:  ", tds[j].X_parent);
      std::cerr << "RBDL:\n" << rbdl.X_lambda[rbdl_j] << std::endl;
      // return false;
    }
  }
  return true;
}

template <typename Algebra>
bool is_equal(const MultiBody<Algebra> &tds,
              const RigidBodyDynamics::Math::VectorNd &q,
              const RigidBodyDynamics::Math::VectorNd &qd,
              const RigidBodyDynamics::Math::VectorNd &qdd) {
  RigidBodyDynamics::Math::VectorNd rbdl_q = q;
  RigidBodyDynamics::Math::VectorNd rbdl_qd = qd;
  RigidBodyDynamics::Math::VectorNd rbdl_qdd = qdd;

  if (tds.is_floating()) {
    // change order to have the dimensions line up for floating-base systems
    rbdl_q[4] = q[0];
    rbdl_q[5] = q[1];
    rbdl_q[6] = q[2];
    rbdl_q[3] = q[3];
    rbdl_q[0] = q[4];
    rbdl_q[1] = q[5];
    rbdl_q[2] = q[6];
    rbdl_qd[3] = qd[0];
    rbdl_qd[4] = qd[1];
    rbdl_qd[5] = qd[2];
    rbdl_qd[0] = qd[3];
    rbdl_qd[1] = qd[4];
    rbdl_qd[2] = qd[5];
    rbdl_qdd[3] = qdd[0];
    rbdl_qdd[4] = qdd[1];
    rbdl_qdd[5] = qdd[2];
    rbdl_qdd[0] = qdd[3];
    rbdl_qdd[1] = qdd[4];
    rbdl_qdd[2] = qdd[5];
  }

  if (!is_equal<Algebra>(tds.q(), rbdl_q)) {
    fprintf(stderr, "Mismatch in q.\n");
    return false;
  }
  if (!is_equal<Algebra>(tds.qd(), rbdl_qd)) {
    fprintf(stderr, "Mismatch in qd.\n");
    return false;
  }
  if (!is_equal<Algebra>(tds.qdd(), rbdl_qdd)) {
    fprintf(stderr, "Mismatch in qdd.\n");
    return false;
  }
  return true;
}
#endif

int main(int argc, char **argv) {
  tds::activate_nan_trap();

  {
    using Algebra = TinyAlgebra<double, DoubleUtils>;

    using Tf = Transform<Algebra>;
    using Vector3 = Algebra::Vector3;
    using VectorX = typename Algebra::VectorX;
    using Matrix3 = Algebra::Matrix3;
    using RigidBodyInertia = RigidBodyInertia<Algebra>;

    Vector3 gravity(0., 0., -9.81);

    UrdfCache<Algebra> cache;
    World<Algebra> world;
    MultiBody<Algebra> *mb = nullptr;

    {
      std::string urdf_filename;
      bool is_floating = false;
      FileUtils::find_file("swimmer/swimmer05/swimmer05.urdf", urdf_filename);
      FileUtils::find_file("pendulum5.urdf", urdf_filename);
      // FileUtils::find_file("sphere8cube.urdf", urdf_filename);
      // FileUtils::find_file("laikago/laikago.urdf", urdf_filename);
      // bool is_floating = true;
      mb = cache.construct(urdf_filename, world, false, is_floating);

      // mb->base_X_world().translation = Algebra::unit3_z();

      mb->qd(2) = 0.5;

      for (std::size_t j = 0; j < mb->size(); ++j) {
        std::cout << "link " << j << ":\n";
        Algebra::print("rbi", (*mb)[j].rbi);
      }
      std::cout << "\n\n\n";
      forward_kinematics(*mb);
      for (std::size_t j = 0; j < mb->size(); ++j) {
        std::cout << "link " << j << ":\n";
        Algebra::print("abi", (*mb)[j].abi);
      }
    }

    // {
    //   mb = new MultiBody<Algebra>;
    //   mb->base_rbi().mass = 0;
    //   mb->base_rbi().com = Algebra::zero3();
    //   mb->base_rbi().inertia = Algebra::zero33();
    //   double mass = 0.5;
    //   Vector3 com(0.0, 0.0, 1.);
    //   Matrix3 I = Algebra::diagonal3(Vector3(1., 1., 1.));
    //   Link<Algebra> link_a(JOINT_REVOLUTE_Y, Tf(0., 0., 1.),
    //                        RigidBodyInertia(mass, com, I));
    //   mb->attach(link_a);
    //   Link<Algebra> link_b(JOINT_REVOLUTE_Y, Tf(0., 0., 1.),
    //                        RigidBodyInertia(mass, com, I));
    //   mb->attach(link_b);
    //   Link<Algebra> link_c(JOINT_REVOLUTE_Y, Tf(0., 0., 1.),
    //                        RigidBodyInertia(mass, com, I));
    //   mb->attach(link_c);
    //   mb->initialize();

    //   mb->q(0) = M_PI_2;
    //   forward_kinematics(*mb);
    // }

#if USE_RBDL
    {
      RigidBodyDynamics::Model rbdl_model = to_rbdl(*mb);
      rbdl_model.gravity = to_rbdl<Algebra>(gravity);

      using VectorND = RigidBodyDynamics::Math::VectorNd;
      VectorND rbdl_q = VectorND::Zero(rbdl_model.q_size);
      VectorND rbdl_qd = VectorND::Zero(rbdl_model.qdot_size);
      VectorND rbdl_qdd = VectorND::Zero(rbdl_model.qdot_size);
      VectorND rbdl_tau = VectorND::Zero(rbdl_model.qdot_size);

      int q_offset = 0, qd_offset = 0;
      if (mb->is_floating()) {
        rbdl_q[0] = Algebra::to_double(mb->q(4));
        rbdl_q[1] = Algebra::to_double(mb->q(5));
        rbdl_q[2] = Algebra::to_double(mb->q(6));
        rbdl_q[3] = Algebra::to_double(mb->q(3));
        rbdl_q[4] = Algebra::to_double(mb->q(0));
        rbdl_q[5] = Algebra::to_double(mb->q(1));
        rbdl_q[6] = Algebra::to_double(mb->q(2));
        q_offset = 7;
        rbdl_qd[0] = Algebra::to_double(mb->qd(3));
        rbdl_qd[1] = Algebra::to_double(mb->qd(4));
        rbdl_qd[2] = Algebra::to_double(mb->qd(5));
        rbdl_qd[3] = Algebra::to_double(mb->qd(0));
        rbdl_qd[4] = Algebra::to_double(mb->qd(1));
        rbdl_qd[5] = Algebra::to_double(mb->qd(2));
        qd_offset = 6;
      }

      for (int i = q_offset; i < mb->dof(); ++i) {
        rbdl_q[i] = Algebra::to_double(mb->q(i));
      }
      for (int i = qd_offset; i < mb->dof_qd(); ++i) {
        rbdl_qd[i] = Algebra::to_double(mb->qd(i));
        rbdl_qdd[i] = Algebra::to_double(mb->qdd(i));
      }
      for (int i = qd_offset; i < mb->dof_actuated() + qd_offset; ++i) {
        rbdl_tau[i] = Algebra::to_double(mb->tau(i - qd_offset));
      }
      RigidBodyDynamics::UpdateKinematics(rbdl_model, rbdl_q, rbdl_qd,
                                          rbdl_qdd);
      if (!is_equal<Algebra>(*mb, rbdl_model)) {
        // exit(1);
      }
      // return 0;

      double dt = 0.001;
      for (int i = 0; i < 500; ++i) {
        printf("\n\n\nt: %i\n", i);
        // forward_kinematics(mb);
        // traj.push_back(mb->q);
        for (auto &link : *mb) {
          Algebra::set_zero(link.a);
        }
        int nd = mb->dof_actuated();
        // Algebra::Index j = 2;
        for (Algebra::Index j = 3; j < nd; ++j) {
          mb->tau(j) = Algebra::sin(i * dt * 10.) * 1e-4;
          rbdl_tau[j] = Algebra::to_double(mb->tau(j));
        }
        forward_dynamics(*mb, gravity);
        for (auto &link : *mb) {
          // Algebra::print(
          //     ("link[" + std::to_string(link.q_index) + "].D").c_str(),
          //     link.D);
          // Algebra::print(
          //     ("link[" + std::to_string(link.q_index) + "].U").c_str(),
          //     link.U);
          // Algebra::print(
          //     ("link[" + std::to_string(link.q_index) + "].S").c_str(),
          //     link.S);
          // Algebra::print(
          //     ("link[" + std::to_string(link.q_index) + "].u").c_str(),
          //     link.u);
          // Algebra::print(
          //     ("TDS link[" + std::to_string(link.q_index) + "].X_world")
          //         .c_str(),
          //     link.X_world);
          // std::cout << "RBDL link[" << link.q_index << "].X_base\n"
          //           << rbdl_model.X_base[link.q_index + 1] << std::endl;
        }
        mb->clear_forces();
        integrate_euler(*mb, dt);

        RigidBodyDynamics::ForwardDynamics(rbdl_model, rbdl_q, rbdl_qd,
                                           rbdl_tau, rbdl_qdd);

        rbdl_qd += rbdl_qdd * dt;
        if (mb->is_floating()) {
          // need to integrate quaternion
          Algebra::Quaternion quat = Algebra::quat_from_xyzw(
              rbdl_q[4], rbdl_q[5], rbdl_q[6], rbdl_q[3]);

          Algebra::Vector3 ang_vel(rbdl_qd[3], rbdl_qd[4], rbdl_qd[5]);
          Algebra::Quaternion dquat = Algebra::quat_velocity(quat, ang_vel, dt);
          quat += dquat;
          Algebra::normalize(quat);
          rbdl_q[4] = Algebra::quat_x(quat);
          rbdl_q[5] = Algebra::quat_y(quat);
          rbdl_q[6] = Algebra::quat_z(quat);
          rbdl_q[3] = Algebra::quat_w(quat);
          // linear velocity integration
          rbdl_q[0] += rbdl_qd[0] * dt;
          rbdl_q[1] += rbdl_qd[1] * dt;
          rbdl_q[2] += rbdl_qd[2] * dt;
          for (int i = 6; i < mb->dof_qd(); ++i) {
            rbdl_q[i + 1] += rbdl_qd[i] * dt;
          }
        } else {
          rbdl_q += rbdl_qd * dt;
        }

        mb->print_state();
        std::cout << "RBDL q: " << rbdl_q.transpose()
                  << "   qd: " << rbdl_qd.transpose()
                  << "   qdd:  " << rbdl_qdd.transpose() << std::endl;

        if (!is_equal<Algebra>(*mb, rbdl_model)) {
          assert(0);
          exit(1);
        }
        if (!is_equal<Algebra>(*mb, rbdl_q, rbdl_qd, rbdl_qdd)) {
          exit(1);
        }

        // compare Jacobians
        tds::forward_kinematics(*mb);
        int jac_link_id = 2;
        Algebra::Vector3 world_point(1., 2., 3.);
        auto tds_jac = tds::point_jacobian(*mb, jac_link_id, world_point);
        RigidBodyDynamics::Math::MatrixNd rbdl_jac(Algebra::num_rows(tds_jac),
                                                   Algebra::num_cols(tds_jac));
        rbdl_jac.setZero();
        // left-associative inverse transform of body_to_world transform
        // (rotation matrix is not inverted)
        Transform<Algebra> link_tf = (*mb)[jac_link_id].X_world;
        Algebra::Vector3 body_point =
            link_tf.rotation * (world_point - link_tf.translation);
        RigidBodyDynamics::CalcPointJacobian(
            rbdl_model, rbdl_q, jac_link_id + 1, to_rbdl<Algebra>(body_point),
            rbdl_jac);

        Algebra::print("TDS Jacobian", tds_jac);
        std::cout << "RBDL Jacobian:\n" << rbdl_jac << std::endl;
        if (!is_equal<Algebra>(tds_jac, rbdl_jac)) {
          exit(1);
        }
      }

      return 0;
    }
#endif

    // {
    //   double mass = .001;
    //   Vector3 com(0., 0., 0.);
    //   double inertial = 0.001;
    //   Matrix3 I = Algebra::diagonal3(Vector3(inertial));
    //   Link<Algebra> link_a(JOINT_PRISMATIC_X, Tf(0., 0., 0.),
    //                        RigidBodyInertia(mass, com, I));
    //   Link<Algebra> link_b(JOINT_PRISMATIC_Y, Tf(0., 0., 0.),
    //                        RigidBodyInertia(mass, com, I));
    //   Link<Algebra> link_c(JOINT_REVOLUTE_Z, Tf(0., 0., 0.),
    //                        RigidBodyInertia(mass, com, I));
    //   Link<Algebra> link_d(JOINT_REVOLUTE_Z, Tf(1., 0., 0.),
    //                        RigidBodyInertia(mass, com, I));
    //   mb = new MultiBody<Algebra>;
    //   mb->attach(link_a);
    //   mb->attach(link_b);
    //   mb->attach(link_c);
    //   mb->attach(link_d);
    // mb->initialize();
    // }

    // return 0;

    // mb->base_rbi = RigidBodyInertia(0.);
    // for (std::size_t j = 2; j < mb.size(); ++j) {
    //   mb[j].rbi = RigidBodyInertia(0.5);
    // }

    // Vector3 gravity(0., 0., -9.81);
    // forward_dynamics(mb, gravity);

    std::vector<typename Algebra::VectorX> traj;

    double dt = 0.001;
    for (int i = 0; i < 50000; ++i) {
      printf("\n\n\nt: %i\n", i);
      // forward_kinematics(mb);
      traj.push_back(mb->q());
      for (auto &link : (*mb)) {
        Algebra::set_zero(link.a);
      }
      int nd = mb->dof_actuated();
      // Algebra::Index j = 2;
      for (Algebra::Index j = 3; j < nd; ++j) {
        mb->tau(j) = Algebra::sin(i * dt * 10.) * 1e-4;
      }
      forward_dynamics(*mb, gravity);
      mb->print_state();
      // for (auto &link : mb) {
      //   Algebra::print(("link[" + std::to_string(link.q_index) +
      //   "].D").c_str(),
      //                  link.D);
      //   Algebra::print(("link[" + std::to_string(link.q_index) +
      //   "].U").c_str(),
      //                  link.U);
      //   Algebra::print(("link[" + std::to_string(link.q_index) +
      //   "].S").c_str(),
      //                  link.S);
      //   Algebra::print(("link[" + std::to_string(link.q_index) +
      //   "].u").c_str(),
      //                  link.u);
      // }
      mb->clear_forces();
      integrate_euler(*mb, dt);
    }

#ifdef USE_MATPLOTLIB
    plot_trajectory<Algebra>(traj, "Trajectory");
#endif
    visualize_trajectory<Algebra>(traj, *mb, dt);
  }

  return 0;

  {
    using Algebra = TinyAlgebra<double, DoubleUtils>;

    using Tf = Transform<Algebra>;
    using Vector3 = Algebra::Vector3;
    using VectorX = typename Algebra::VectorX;
    using Matrix3 = Algebra::Matrix3;
    using RigidBodyInertia = RigidBodyInertia<Algebra>;

    MultiBody<Algebra> mb;
    double mass = 1.;
    Vector3 com(0., 0., 1.);
    Matrix3 I = Algebra::diagonal3(Vector3(1., 1., 1.));
    Link<Algebra> link_a(JOINT_REVOLUTE_Y, Tf(0., 0., 1.),
                         RigidBodyInertia(mass, com, I));
    Link<Algebra> link_b(JOINT_REVOLUTE_Y, Tf(0., 0., 1.),
                         RigidBodyInertia(mass, com, I));
    mb.attach(link_a);
    mb.attach(link_b);
    mb.initialize();

    mb.q() = VectorX({M_PI_2, 0.0});

    forward_kinematics(mb);
    Vector3 gravity(0., 0., -9.81);
    forward_dynamics(mb, gravity);

    std::vector<typename Algebra::VectorX> traj;

    double dt = 0.01;
    for (int i = 0; i < 1000; ++i) {
      traj.push_back(mb.q());
      integrate_euler(mb, dt);
      forward_dynamics(mb, gravity);
      mb.print_state();
    }

    // plot_trajectory<Algebra>(traj);
    visualize_trajectory<Algebra>(traj, mb, dt, 1);
  }

  return 0;
}