#include <stdio.h>

#include <cassert>
//#include "stan_double_utils.h"

#include <ceres/autodiff_cost_function.h>

#include <chrono>  // std::chrono::seconds
#include <thread>  // std::this_thread::sleep_for

#include "math/enoki_algebra.hpp"
#include "math/pose.hpp"
#include "math/tiny/ceres_utils.h"
#include "math/tiny/tiny_algebra.hpp"
#include "math/tiny/tiny_double_utils.h"
#include "math/tiny/tiny_dual.h"
#include "math/tiny/tiny_dual_double_utils.h"
#include "pybullet_visualizer_api.h"
#include "rigid_body.hpp"
#include "utils/file_utils.hpp"
#include "world.hpp"

typedef PyBulletVisualizerAPI VisualizerAPI;
std::string sphere2red;

VisualizerAPI* visualizer = nullptr;

// ID of the ball whose position is optimized for
const int TARGET_ID = 0;
constexpr int steps = 50;

using namespace tds;

template <typename Algebra>
typename Algebra::Scalar rollout(const typename Algebra::Scalar* force_y) {
  typename Algebra::Scalar dt = Algebra::fraction(1, 20);
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

  //Create target ball
  Scalar x = Algebra::zero(), y = Algebra::zero();
  const Geometry* geom = world.create_sphere(radius);
  RigidBody* body = world.create_rigid_body(mass, geom);
  body->world_pose().position = Vector3(x, y, Algebra::zero());
  bodies.push_back(body);

  // Create white ball
  const Geometry* white_geom = world.create_sphere(radius);
  RigidBody* white_ball = world.create_rigid_body(mass, white_geom);
  white_ball->world_pose().position = Vector3(Algebra::zero(), -Algebra::two(), Algebra::zero());
  bodies.push_back(white_ball);

  for (int i = 0; i < steps; i++) {
    white_ball->apply_central_force(Vector3(Algebra::zero(), force_y[i], Algebra::zero()));
    world.step(dt);
  }

  // Compute error
  Vector3 delta = bodies[TARGET_ID]->world_pose().position - target;
  return Algebra::sqnorm(delta);
}


template <typename Algebra>
void visualize_trajectory(const typename Algebra::Scalar* force_y, VisualizerAPI* vis = nullptr) {
  typename Algebra::Scalar dt = Algebra::fraction(1, 10);
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

  //Create target ball
  // Scalar deg_60 = Algebra::pi() / Algebra::fraction(3, 1);  // even triangle
  // Scalar dx = Algebra::cos(deg_60) * radius * Algebra::two();
  // Scalar dy = Algebra::sin(deg_60) * radius * Algebra::two();
  Scalar x = Algebra::zero(), y = Algebra::zero();
  int ball_id = 0;
  // for (int column = 1; column <= 3; ++column) 
  {
    int column = 1;
    // Scalar x = rx;
    // for (int i = 0; i < column; ++i) 
    {
      int i = 0;
      const Geometry* geom = world.create_sphere(radius);
      RigidBody* body = world.create_rigid_body(mass, geom);
      body->world_pose().position = Vector3(x, y, Algebra::zero());
      bodies.push_back(body);
      b3RobotSimulatorLoadUrdfFileArgs args;
      args.m_startPosition.setX(Algebra::to_double(x));
      args.m_startPosition.setY(Algebra::to_double(y));
      int sphere_id = visualizer->loadURDF(sphere2red, args);
      visuals.push_back(sphere_id);
      if (ball_id == TARGET_ID) {
        b3RobotSimulatorChangeVisualShapeArgs vargs;
        vargs.m_objectUniqueId = sphere_id;
        vargs.m_hasRgbaColor = true;
        vargs.m_rgbaColor = btVector4(0, 0.6, 1, 1);
        vis->changeVisualShape(vargs);
      }
      // ++ball_id;
      // x += radius * Algebra::two();
    }
    // rx = rx - dx;
    // y = y + dy;
  }

  // Create white ball
  Vector3 white = Vector3(Algebra::zero(), -Algebra::two(), Algebra::zero());
  const Geometry* white_geom = world.create_sphere(radius);
  RigidBody* white_ball = world.create_rigid_body(mass, white_geom);
  white_ball->world_pose().position = Vector3(white.x(), white.y(), white.z());
  bodies.push_back(white_ball);

  {
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

  for (int i = 0; i < steps; i++) {
    visualizer->submitProfileTiming("world.step");
    white_ball->apply_central_force(Vector3(Algebra::zero(), force_y[i], Algebra::zero()));
    world.step(dt);
    visualizer->submitProfileTiming("");

    {
      double dtd = Algebra::to_double(dt);
      // update visualization
      std::this_thread::sleep_for(std::chrono::duration<double>(dtd));
      for (int b = 0; b < bodies.size(); b++) {
        const RigidBody* body = bodies[b];
        int sphere_id = visuals[b];
        btVector3 base_pos(Algebra::to_double(body->world_pose().position[0]),
                           Algebra::to_double(body->world_pose().position[1]),
                           Algebra::to_double(body->world_pose().position[2]));
        btQuaternion base_orn(
            Algebra::to_double(Algebra::quat_x(body->world_pose().orientation)),
            Algebra::to_double(Algebra::quat_y(body->world_pose().orientation)),
            Algebra::to_double(Algebra::quat_z(body->world_pose().orientation)),
            Algebra::to_double(
                Algebra::quat_w(body->world_pose().orientation)));
        visualizer->resetBasePositionAndOrientation(sphere_id, base_pos,
                                                    base_orn);
      }
    }
  }
}


struct CeresFunctional {
  // int steps{steps};

  template <typename T>
  bool operator()(const T* f, T* e) const {
    typedef ceres::Jet<double, steps> Jet;
    typedef std::conditional_t<std::is_same_v<T, double>, DoubleUtils,
                               CeresUtils<steps>>
        Utils;
    *e = rollout<TinyAlgebra<T, Utils>>(f);
    return true;
  }
};

ceres::AutoDiffCostFunction<CeresFunctional, 1, steps> cost_function(
    new CeresFunctional);
double* parameters = new double[steps];
double* gradient = new double[steps];

void grad_ceres(double* force_y, double* cost,
                double* d_force_y) {
  double const* const* params = &force_y;
  cost_function.Evaluate(params, cost, &gradient);
  d_force_y = gradient;
}

int main(int argc, char* argv[]) {
  tds::FileUtils::find_file("sphere2red.urdf", sphere2red);
  std::string connection_mode = "shared_memory"; //requires pybullet server running in background

  using namespace std::chrono;

  visualizer = new VisualizerAPI;
  visualizer->setTimeOut(1e30);
  printf("\nmode=%s\n", const_cast<char*>(connection_mode.c_str()));
  int mode = eCONNECT_GUI;
  if (connection_mode == "direct") mode = eCONNECT_DIRECT;
  if (connection_mode == "shared_memory") mode = eCONNECT_SHARED_MEMORY;
  visualizer->connect(mode);

  visualizer->resetSimulation();

  double init_force_y = 1.;
  using DoubleAlgebra = TinyAlgebra<double, DoubleUtils>;
  
  {
    double cost;
    std::vector<double> d_force_y(steps, 0.0);
    std::vector<double> force_y(steps, init_force_y);
    double learning_rate = 1e2;
    for (int iter = 0; iter < 50; ++iter) {
      grad_ceres(force_y.data(), &cost, d_force_y.data());
      printf("Iteration %02d - cost: %.3f \n", iter, cost);
      
      for (size_t i = 0; i < steps; i++)
      {
        force_y[i] -= learning_rate * d_force_y[i];
      }
    }
    // Print decision variables (force_y)
    printf("Final force: [");
    for (auto i: force_y)
      printf("%.2f, ", i);
    std::cout<<"]\n";

    fflush(stdout);
    rollout<DoubleAlgebra>(force_y.data());
    visualize_trajectory<DoubleAlgebra>(force_y.data(), visualizer);
  }

  visualizer->disconnect();
  delete visualizer;

  return 0;
}
