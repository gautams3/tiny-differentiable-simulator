#include <stdio.h>
#include <cassert>
#include <ceres/autodiff_cost_function.h>
#include <chrono>  // std::chrono::seconds
#include <thread>  // std::this_thread::sleep_for

#include "math/pose.hpp"
#include "math/tiny/ceres_utils.h"
#include "math/tiny/tiny_algebra.hpp"
#include "math/tiny/tiny_double_utils.h"
#include "pybullet_visualizer_api.h"
#include "rigid_body.hpp"
#include "utils/file_utils.hpp"
#include "world.hpp"

typedef PyBulletVisualizerAPI VisualizerAPI;
std::string sphere2red;

VisualizerAPI* visualizer = nullptr;

const int TARGET_ID = 0;  // ID of the ball whose position is optimized for
constexpr int steps = 50;
using DoubleAlgebra = TinyAlgebra<double, DoubleUtils>;
constexpr DoubleAlgebra::Scalar fps = 20;

using namespace tds;

template <typename Algebra>
// forcing 'states' to be of type double, not part of decision variables
// TODO: What's the preferred design here
typename Algebra::Scalar rollout(
    const typename Algebra::Scalar* force_y,
    std::vector<std::vector<DoubleAlgebra::Vector3>>& states) {
  typename Algebra::Scalar dt = Algebra::fraction(
      1, fps);  // TODO: want to specify dtd as double instead of fps as int
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  typedef tds::RigidBody<Algebra> RigidBody;
  typedef tds::Geometry<Algebra> Geometry;

  Vector3 target(Algebra::zero(), Algebra::two(), Algebra::zero());
  Scalar gravity_z = Algebra::zero();
  tds::World<Algebra> world(gravity_z);

  std::vector<RigidBody*> bodies;

  Scalar radius = Algebra::half();
  Scalar mass = Algebra::one();

  // Create target ball
  Scalar x = Algebra::zero(), y = Algebra::zero();
  const Geometry* geom = world.create_sphere(radius);
  RigidBody* body = world.create_rigid_body(mass, geom);
  body->world_pose().position = Vector3(x, y, Algebra::zero());
  bodies.push_back(body);

  // Create white ball
  const Geometry* white_geom = world.create_sphere(radius);
  RigidBody* white_ball = world.create_rigid_body(mass, white_geom);
  white_ball->world_pose().position =
      Vector3(Algebra::zero(), -Algebra::two(), Algebra::zero());
  bodies.push_back(white_ball);

  for (int i = 0; i < steps; i++) {
    white_ball->apply_central_force(
        Vector3(Algebra::zero(), force_y[i], Algebra::zero()));
    world.step(dt);
    // Store states
    for (int b = 0; b < bodies.size(); b++) {
      const RigidBody* body = bodies[b];
      states[i][b][0] = Algebra::to_double(body->world_pose().position[0]);
      states[i][b][1] = Algebra::to_double(body->world_pose().position[1]);
      states[i][b][2] = Algebra::to_double(body->world_pose().position[2]);
    }
  }

  // Compute error
  Vector3 delta = bodies[TARGET_ID]->world_pose().position - target;
  return Algebra::sqnorm(delta);
}

template <typename Algebra>
void visualize_trajectory(
    const typename Algebra::Scalar* force_y,
    std::vector<std::vector<DoubleAlgebra::Vector3>>& states,
    VisualizerAPI* vis = nullptr) {
  typename Algebra::Scalar dt = Algebra::fraction(1, fps);
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  typedef tds::RigidBody<Algebra> RigidBody;
  typedef tds::Geometry<Algebra> Geometry;

  std::vector<int> visuals;
  Vector3 target(Algebra::zero(), Algebra::two(), Algebra::zero());
  vis->resetSimulation();

  Scalar gravity_z = Algebra::zero();
  tds::World<Algebra> world(gravity_z);

  std::vector<RigidBody*> bodies;

  Scalar radius = Algebra::half();
  Scalar mass = Algebra::one();
  Scalar x = Algebra::zero(), y = Algebra::zero();

  {
    // Create target ball
    const Geometry* geom = world.create_sphere(radius);
    RigidBody* body = world.create_rigid_body(mass, geom);
    body->world_pose().position = Vector3(x, y, Algebra::zero());
    bodies.push_back(body);
    b3RobotSimulatorLoadUrdfFileArgs args;
    args.m_startPosition.setX(Algebra::to_double(x));
    args.m_startPosition.setY(Algebra::to_double(y));
    int sphere_id = visualizer->loadURDF(sphere2red, args);
    visuals.push_back(sphere_id);

    // change colour
    b3RobotSimulatorChangeVisualShapeArgs vargs;
    vargs.m_objectUniqueId = sphere_id;
    vargs.m_hasRgbaColor = true;
    vargs.m_rgbaColor = btVector4(0, 0.6, 1, 1);
    vis->changeVisualShape(vargs);
  }

  {
    // Create white ball
    Vector3 white = Vector3(Algebra::zero(), -Algebra::two(), Algebra::zero());
    const Geometry* white_geom = world.create_sphere(radius);
    RigidBody* white_ball = world.create_rigid_body(mass, white_geom);
    white_ball->world_pose().position =
        Vector3(white.x(), white.y(), white.z());
    bodies.push_back(white_ball);

    // visualize white ball
    b3RobotSimulatorLoadUrdfFileArgs args;
    args.m_startPosition.setX(Algebra::to_double(white.x()));
    args.m_startPosition.setY(Algebra::to_double(white.y()));
    args.m_startPosition.setZ(Algebra::to_double(white.z()));
    int sphere_id = visualizer->loadURDF(sphere2red, args);
    visuals.push_back(sphere_id);
    b3RobotSimulatorChangeVisualShapeArgs vargs;
    vargs.m_objectUniqueId = sphere_id;
    vargs.m_hasRgbaColor = true;
    vargs.m_rgbaColor = btVector4(1, 1, 1, 1);
    vis->changeVisualShape(vargs);
  }

  {
    // visualize target
    b3RobotSimulatorLoadUrdfFileArgs args;
    args.m_startPosition.setX(Algebra::to_double(target.x()));
    args.m_startPosition.setY(Algebra::to_double(target.y()));
    args.m_startPosition.setZ(Algebra::to_double(target.z()));
    int sphere_id = visualizer->loadURDF(sphere2red, args);
    visuals.push_back(sphere_id);
    b3RobotSimulatorChangeVisualShapeArgs vargs;
    vargs.m_objectUniqueId = sphere_id;
    vargs.m_hasRgbaColor = true;
    vargs.m_rgbaColor = btVector4(1, 0.6, 0, 0.8);
    vis->changeVisualShape(vargs);
  }

  // Visualization over time
  double dtd = Algebra::to_double(dt);
  for (int i = 0; i < steps; i++) {
    std::this_thread::sleep_for(std::chrono::duration<double>(dtd));
    for (int b = 0; b < bodies.size(); b++) {
      const RigidBody* body = bodies[b];
      int sphere_id = visuals[b];
      btVector3 base_pos(states[i][b][0], states[i][b][1], states[i][b][2]);
      // fix quat. ASSUMING ALL SPHERES
      btQuaternion base_orn(0.0, 0.0, 0.0, 1.0);
      visualizer->resetBasePositionAndOrientation(sphere_id, base_pos,
                                                  base_orn);
    }
  }
}

struct CeresFunctional {
  template <typename T>
  bool operator()(const T* f, T* e) const {
    typedef ceres::Jet<double, steps> Jet;
    typedef std::conditional_t<std::is_same_v<T, double>, DoubleUtils,
                               CeresUtils<steps>>
        Utils;
    DoubleAlgebra::Vector3 init_posn(0.0, 0.0, 0.0);
    // 2 = num_bodies. TODO How to make bodies.size() global?
    std::vector<DoubleAlgebra::Vector3> init_state(2, init_posn);
    std::vector<std::vector<DoubleAlgebra::Vector3>> dummy_states(steps,
                                                                  init_state);
    *e = rollout<TinyAlgebra<T, Utils>>(f, dummy_states);
    return true;
  }
};

ceres::AutoDiffCostFunction<CeresFunctional, 1, steps> cost_function(
    new CeresFunctional);

void grad_ceres(double* force_y, double* cost, double* d_force_y) {
  double const* const* params = &force_y;
  cost_function.Evaluate(params, cost, &d_force_y);
}

int main(int argc, char* argv[]) {
  tds::FileUtils::find_file("sphere2red.urdf", sphere2red);
  // requires pybullet server running in background
  // TODO: direct mode not working on my machine
  std::string connection_mode = "shared_memory";

  using namespace std::chrono;

  visualizer = new VisualizerAPI;
  visualizer->setTimeOut(1e30);
  printf("\nmode=%s\n", const_cast<char*>(connection_mode.c_str()));
  int mode = eCONNECT_GUI;
  if (connection_mode == "direct") mode = eCONNECT_DIRECT;
  if (connection_mode == "shared_memory") mode = eCONNECT_SHARED_MEMORY;
  visualizer->connect(mode);

  visualizer->resetSimulation();

  double init_force_y = 3.;
  DoubleAlgebra::Vector3 init_posn(0.0, 0.0, 0.0);
  std::vector<DoubleAlgebra::Vector3> init_state(
      2, init_posn);  // magic number 2 = num_bodies
  std::vector<std::vector<DoubleAlgebra::Vector3>> states(steps, init_state);
  double cost;
  std::vector<double> d_force_y(steps, 0.0);
   // TODO: how to use DoubleAlgebra::VectorX here
  std::vector<double> force_y(steps, init_force_y);
  double learning_rate = 1e0;

  // Before optimization
  printf("ROLLOUT BEFORE OPTIM (see bullet)");
  rollout<DoubleAlgebra>(force_y.data(), states);
  visualize_trajectory<DoubleAlgebra>(force_y.data(), states, visualizer);

  for (int iter = 0; iter < 50; ++iter) {
    grad_ceres(force_y.data(), &cost, d_force_y.data());
    printf("Iteration %02d - cost: %.3f \n", iter, cost);

    for (size_t i = 0; i < steps; i++) {
      force_y[i] -= learning_rate * d_force_y[i];
    }
  }

  // Print decision variables (force_y)
  // TODO: use DoubleAlgebra::print() after force_y is type VectorX
  printf("Final force: [");
  for (auto i : force_y) printf("%.2f, ", i);
  std::cout << "]\n";

  fflush(stdout);

  // After optimization
  printf("ROLLOUT AFTER OPTIM (see bullet)");
  rollout<DoubleAlgebra>(force_y.data(), states);
  visualize_trajectory<DoubleAlgebra>(force_y.data(), states, visualizer);

  visualizer->disconnect();
  delete visualizer;

  return 0;
}
