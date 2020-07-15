// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>

#include "assert.h"
#include "tiny_world.h"
//#include "stan_double_utils.h"
#include "tiny_dual.h"
#include "tiny_matrix3x3.h"
#include "tiny_quaternion.h"
#include "tiny_vector3.h"

#include <ceres/autodiff_cost_function.h>

#include <chrono>  // std::chrono::seconds
#include <thread>  // std::this_thread::sleep_for

#include "ceres_utils.h"
#include "tiny_double_utils.h"
#include "tiny_dual_double_utils.h"
#include "tiny_multi_body.h"
#include "tiny_pose.h"
#include "tiny_rigid_body.h"

#include "pybullet_visualizer_api.h"
#include "tiny_file_utils.h"

typedef PyBulletVisualizerAPI VisualizerAPI;
std::string box;
std::string sphere2red;

VisualizerAPI* visualizer = nullptr;

// ID of the ball whose position is optimized for
const int TARGET_ID = 0;

template <typename TinyScalar, typename TinyConstants>
TinyScalar rollout(TinyScalar force_x, TinyScalar force_y, int steps = 300,
                   VisualizerAPI* vis = nullptr,
                   TinyScalar dt = TinyConstants::fraction(1, 60)) {
  typedef TinyVector3<TinyScalar, TinyConstants> TinyVector3;
  typedef TinyRigidBody<TinyScalar, TinyConstants> TinyRigidBody;
  typedef TinyGeometry<TinyScalar, TinyConstants> TinyGeometry;

  std::vector<int> visuals;
  TinyVector3 target(TinyConstants::fraction(0, 10),
                     TinyConstants::fraction(8, 1), TinyConstants::zero());
  if (vis) {
    vis->resetSimulation();
  }

  TinyScalar gravity_z = TinyConstants::zero();
  TinyWorld<TinyScalar, TinyConstants> world(gravity_z);

  std::vector<TinyRigidBody*> bodies;

  TinyScalar length = TinyConstants::one();
  TinyScalar mass = TinyConstants::one();
  TinyScalar x = TinyConstants::zero(), y = TinyConstants::zero();
  
  const TinyGeometry* geom = world.create_box(length);
  TinyRigidBody* body = world.create_rigid_body(mass, geom);
  body->m_world_pose.m_position =
      TinyVector3::create(x, y, TinyConstants::zero());
  bodies.push_back(body);
  if (vis) {
    b3RobotSimulatorLoadUrdfFileArgs args;
    args.m_startPosition.setX(TinyConstants::getDouble(x));
    args.m_startPosition.setY(TinyConstants::getDouble(y));
    int box_id = visualizer->loadURDF(box, args);
    visuals.push_back(box_id);
    b3RobotSimulatorChangeVisualShapeArgs vargs;
    vargs.m_objectUniqueId = box_id;
    vargs.m_hasRgbaColor = true;
    vargs.m_rgbaColor = btVector4(0, 0.6, 1, 1);
    vis->changeVisualShape(vargs);
  }

  // Create white ball
  TinyScalar radius = TinyConstants::half();
  TinyVector3 white = TinyVector3::create(
      TinyConstants::zero(), -TinyConstants::two(), TinyConstants::zero());
  const TinyGeometry* white_geom = world.create_sphere(radius);
  TinyRigidBody* white_ball = world.create_rigid_body(mass, white_geom);
  white_ball->m_world_pose.m_position =
      TinyVector3::create(white.x(), white.y(), white.z());
  bodies.push_back(white_ball);
  white_ball->apply_central_force(
      TinyVector3::create(force_x, force_y, TinyConstants::zero()));

  
  if (vis) {
    {
      // visualize white ball
      b3RobotSimulatorLoadUrdfFileArgs args;
      args.m_startPosition.setX(TinyConstants::getDouble(white.x()));
      args.m_startPosition.setY(TinyConstants::getDouble(white.y()));
      args.m_startPosition.setZ(TinyConstants::getDouble(white.z()));
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
      args.m_startPosition.setX(TinyConstants::getDouble(target.x()));
      args.m_startPosition.setY(TinyConstants::getDouble(target.y()));
      args.m_startPosition.setZ(TinyConstants::getDouble(target.z()));
      int box_id = visualizer->loadURDF(box, args);
      visuals.push_back(box_id);
      b3RobotSimulatorChangeVisualShapeArgs vargs;
      vargs.m_objectUniqueId = box_id;
      vargs.m_hasRgbaColor = true;
      vargs.m_rgbaColor = btVector4(1, 0.6, 0, 0.8);
      vis->changeVisualShape(vargs);
    }
  }

  for (int i = 0; i < steps; i++) {
    visualizer->submitProfileTiming("world.step");
    world.step(dt);
    visualizer->submitProfileTiming("");

    if (vis) {
      double dtd = TinyConstants::getDouble(dt);
      // update visualization
      std::this_thread::sleep_for(std::chrono::duration<double>(dtd));
      for (int b = 0; b < bodies.size(); b++) {
        const TinyRigidBody* body = bodies[b];
        int box_id = visuals[b];
        btVector3 base_pos(
            TinyConstants::getDouble(body->m_world_pose.m_position.getX()),
            TinyConstants::getDouble(body->m_world_pose.m_position.getY()),
            TinyConstants::getDouble(body->m_world_pose.m_position.getZ()));
        btQuaternion base_orn(
            TinyConstants::getDouble(body->m_world_pose.m_orientation.getX()),
            TinyConstants::getDouble(body->m_world_pose.m_orientation.getY()),
            TinyConstants::getDouble(body->m_world_pose.m_orientation.getZ()),
            TinyConstants::getDouble(body->m_world_pose.m_orientation.getW()));
        visualizer->resetBasePositionAndOrientation(box_id, base_pos,
                                                    base_orn);
      }
    }
  }

  // Compute error
  TinyVector3 delta = bodies[TARGET_ID]->m_world_pose.m_position - target;
  return delta.sqnorm();
}

struct CeresFunctional {
  int steps{300};

  template <typename T>
  bool operator()(const T* const x, T* e) const {
    typedef ceres::Jet<double, 2> Jet;
    T fx(x[0]), fy(x[1]);
    typedef std::conditional_t<std::is_same_v<T, double>, DoubleUtils,
                               CeresUtils<2>>
        Utils;
    *e = rollout<T, Utils>(fx, fy, steps);
    return true;
  }
};

ceres::AutoDiffCostFunction<CeresFunctional, 1, 2> cost_function(
    new CeresFunctional);
double* parameters = new double[2];
double* gradient = new double[2];

void grad_ceres(double force_x, double force_y, double* cost, double* d_force_x,
                double* d_force_y, int steps = 300) {
  parameters[0] = force_x;
  parameters[1] = force_y;
  double const* const* params = &parameters;
  cost_function.Evaluate(params, cost, &gradient);
  *d_force_x = gradient[0];
  *d_force_y = gradient[1];
}

int main(int argc, char* argv[]) {
  // TinyFileUtils::find_file("sphere2red.urdf", sphere2red);
  TinyFileUtils::find_file("sphere2red.urdf", sphere2red);
  TinyFileUtils::find_file("sphere8cube.urdf", box);
  std::string connection_mode = "gui";

  using namespace std::chrono;

  visualizer = new VisualizerAPI;
  visualizer->setTimeOut(1e30);
  printf("mode=%s\n", const_cast<char*>(connection_mode.c_str()));
  int mode = eCONNECT_GUI;
  if (connection_mode == "direct") mode = eCONNECT_DIRECT;
  if (connection_mode == "shared_memory") mode = eCONNECT_SHARED_MEMORY;
  mode = eCONNECT_SHARED_MEMORY; //force shared memory (requires pybullet server to be running in the background)
  visualizer->connect(mode);

  visualizer->resetSimulation();

  double init_force_x = 0., init_force_y = 500.;
  int steps = 300;
  int gd_steps = 100; // gradient_descent_steps
  double learning_rate = 3e1;
  rollout<double, DoubleUtils>(init_force_x, init_force_y, steps, visualizer);

  {
    auto start = high_resolution_clock::now();
    double cost, d_force_x, d_force_y;
    // double learning_rate = 1e2;
    double force_x = init_force_x, force_y = init_force_y;
    for (int iter = 0; iter < gd_steps; ++iter) {
      grad_ceres(force_x, force_y, &cost, &d_force_x, &d_force_y, steps);
      printf("Iteration %02d - cost: %.3f \tforce: [%.2f %2.f]\n", iter, cost,
             force_x, force_y);
      force_x -= learning_rate * d_force_x;
      force_y -= learning_rate * d_force_y;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("Ceres Jet took %ld microseconds.\n",
           static_cast<long>(duration.count()));
    rollout<double, DoubleUtils>(force_x, force_y, steps, visualizer);
  }

  visualizer->disconnect();
  delete visualizer;

  return 0;
}
