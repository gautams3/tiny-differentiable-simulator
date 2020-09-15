#include <stdio.h>
#include <cassert>
#include "ceres/ceres.h"
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
constexpr int kSteps = 50;  // global specification of steps
using DoubleAlgebra = TinyAlgebra<double, DoubleUtils>;
constexpr DoubleAlgebra::Scalar kfps = 20;  // global fps
constexpr int kNumBodies = 2;  // number of bodies in this experiment

using namespace tds;

template <typename Algebra>
// forcing 'states' to be of type double, not part of decision variables
typename Algebra::Scalar rollout(
    const typename Algebra::Scalar* force_y,
    std::vector<std::vector<DoubleAlgebra::Vector3>>& states,
    size_t steps) {
  typename Algebra::Scalar dt = Algebra::fraction(1, kfps);
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  typedef tds::RigidBody<Algebra> RigidBody;
  typedef tds::Geometry<Algebra> Geometry;

  Scalar radius = Algebra::half();
  Scalar mass = Algebra::one();

  //Goal location
  Vector3 target(Algebra::zero(), Algebra::two(), radius);
  Scalar gravity_z = -Algebra::fraction(981, 100);
  tds::World<Algebra> world(gravity_z);

  std::vector<RigidBody*> bodies;

  // Create target ball
  Scalar x = Algebra::zero(), y = Algebra::zero();
  const Geometry* geom = world.create_sphere(radius);
  RigidBody* body = world.create_rigid_body(mass, geom);
  body->world_pose().position = Vector3(x, y, radius);
  bodies.push_back(body);

  // Create white ball
  const Geometry* white_geom = world.create_sphere(radius);
  RigidBody* white_ball = world.create_rigid_body(mass, white_geom);
  white_ball->world_pose().position =
      Vector3(Algebra::zero(), -Algebra::two(), radius);
  bodies.push_back(white_ball);

  assert(("Optim problem setup assuming kNumBodies number of bodies",
          bodies.size() == kNumBodies));

  {
    Scalar mass = Scalar(0.0);
    std::string filename;
    tds::FileUtils::find_file("plane_implicit.urdf", filename);
    int plane_id = visualizer->loadURDF(filename);
    const tds::Geometry<Algebra>* geom = world.create_plane();
    tds::RigidBody<Algebra>* body = world.create_rigid_body(mass, geom);
    // bodies.push_back(body);
    // visuals.push_back(plane_id);
  }

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
    size_t steps,
    VisualizerAPI* vis = nullptr) {
  typename Algebra::Scalar dt = Algebra::fraction(1, kfps);
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  typedef tds::RigidBody<Algebra> RigidBody;
  typedef tds::Geometry<Algebra> Geometry;

  std::vector<int> visuals;
  Vector3 target(Algebra::zero(), Algebra::two(), Algebra::fraction(1, 2));
  vis->resetSimulation();

  Scalar gravity_z = -Algebra::fraction(981, 100);
  tds::World<Algebra> world(gravity_z);

  std::vector<RigidBody*> bodies;

  Scalar radius = Algebra::half();
  Scalar mass = Algebra::one();
  Scalar x = Algebra::zero(), y = Algebra::zero();

  {
    // Create target ball
    const Geometry* geom = world.create_sphere(radius);
    RigidBody* body = world.create_rigid_body(mass, geom);
    body->world_pose().position = Vector3(x, y, radius);
    bodies.push_back(body);
    b3RobotSimulatorLoadUrdfFileArgs args;
    args.m_startPosition.setX(Algebra::to_double(x));
    args.m_startPosition.setY(Algebra::to_double(y));
    args.m_startPosition.setZ(Algebra::to_double(radius));
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
    Vector3 white = Vector3(Algebra::zero(), -Algebra::two(), radius);
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

  //visualize plane
  {
  Scalar mass = Scalar(0.0);
  std::string filename;
  tds::FileUtils::find_file("plane_implicit.urdf", filename);
  int plane_id = visualizer->loadURDF(filename);
  const tds::Geometry<Algebra>* geom = world.create_plane();
  tds::RigidBody<Algebra>* body = world.create_rigid_body(mass, geom);
  // bodies.push_back(body);
  visuals.push_back(plane_id);
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
  bool operator()(const T* force, T* cost) const {
    typedef ceres::Jet<double, kSteps> Jet;
    typedef std::conditional_t<std::is_same_v<T, double>, DoubleUtils,
                               CeresUtils<kSteps>>
        Utils;
    DoubleAlgebra::Vector3 init_posn(0.0, 0.0, 0.0);
    std::vector<DoubleAlgebra::Vector3> init_state(kNumBodies, init_posn);
    std::vector<std::vector<DoubleAlgebra::Vector3>> dummy_states(kSteps,
                                                                  init_state);
    *cost = rollout<TinyAlgebra<T, Utils>>(force, dummy_states, kSteps);
    return true;
  }
};

void print_trajectory(const std::vector<std::vector<DoubleAlgebra::Vector3>> &states, const std::vector<double> &inputs, size_t steps) {
  printf("yb = y coord of target (blue); yw = y coord of 'robot' white; f = force applied to 'robot'\n");
  for (size_t i = 0; i < steps; i++) {
    printf("%lu: yb %.3f\tyw %.3f\tf %.3f\n", i, states[i][TARGET_ID][1], states[i][1][1], inputs[i]);
  }
}

int main(int argc, char* argv[]) {
  tds::FileUtils::find_file("sphere2red.urdf", sphere2red);
  // 'shared_memory' requires pybullet server running in background
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

  //Initialize variables
  double init_force_y = 3.;
  DoubleAlgebra::Vector3 init_posn(0.0, 0.0, 0.0);
  std::vector<DoubleAlgebra::Vector3> init_state(kNumBodies, init_posn);
  std::vector<std::vector<DoubleAlgebra::Vector3>> states(kSteps, init_state);
  double cost;
  std::vector<double> force_y(kSteps, init_force_y);

  ceres::Problem problem;
  ceres::AutoDiffCostFunction<CeresFunctional, 1, kSteps> cost_function(
      new CeresFunctional);
  problem.AddResidualBlock(&cost_function, NULL, force_y.data());
  double max_force = 10.0;
  for (size_t i = 0; i < force_y.size(); i++)
  {
    problem.SetParameterLowerBound(force_y.data(), i, 0.0);
    problem.SetParameterUpperBound(force_y.data(), i, max_force);
  }

  // Before optimization
  printf("ROLLOUT BEFORE OPTIM (see bullet)\n");
  rollout<DoubleAlgebra>(force_y.data(), states, kSteps);
  visualize_trajectory<DoubleAlgebra>(force_y.data(), states, kSteps,
                                      visualizer);

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 500;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";

  // After optimization
  printf("ROLLOUT AFTER OPTIM (see bullet)\n");
  rollout<DoubleAlgebra>(force_y.data(), states, kSteps);
  print_trajectory(states, force_y, kSteps);
  visualize_trajectory<DoubleAlgebra>(force_y.data(), states, kSteps,
                                      visualizer);

  visualizer->disconnect();
  delete visualizer;

  return 0;
}
