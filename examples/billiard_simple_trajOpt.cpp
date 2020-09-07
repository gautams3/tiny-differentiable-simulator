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
std::string sphere2red;

VisualizerAPI* visualizer = nullptr;

// ID of the ball whose position is optimized for
const int TARGET_ID = 0;

template <typename TinyScalar, typename TinyConstants>
TinyScalar rollout(std::vector<TinyScalar> force_x, int steps = 20) {
  TinyScalar dt = TinyConstants::fraction(1, steps);
  typedef TinyVector3<TinyScalar, TinyConstants> TinyVector3;
  typedef TinyRigidBody<TinyScalar, TinyConstants> TinyRigidBody;
  typedef TinyGeometry<TinyScalar, TinyConstants> TinyGeometry;

  TinyVector3 goal(TinyConstants::one(), TinyConstants::zero(), TinyConstants::zero());

  TinyScalar gravity_z = TinyConstants::zero();
  TinyWorld<TinyScalar, TinyConstants> world(gravity_z);

  std::vector<TinyRigidBody*> bodies;

  TinyScalar radius = TinyConstants::half();
  TinyScalar mass = TinyConstants::one();
  TinyScalar x = TinyConstants::zero(), y = TinyConstants::zero();

  // Create target ball
  const TinyGeometry* geom = world.create_sphere(radius);
  TinyRigidBody* target_ball = world.create_rigid_body(mass, geom);
  target_ball->m_world_pose.m_position = TinyVector3::create(x, y, TinyConstants::zero());
  bodies.push_back(target_ball);

  // Create white ball
  const TinyGeometry* white_geom = world.create_sphere(radius);
  TinyRigidBody* white_ball = world.create_rigid_body(mass, white_geom);
  white_ball->m_world_pose.m_position = TinyVector3::create(-TinyConstants::two(), TinyConstants::zero(), TinyConstants::zero());
  bodies.push_back(white_ball);

  for (int i = 0; i < steps; i++) {
    //Apply force
    white_ball->apply_central_force(
      TinyVector3::create(force_x[i], TinyConstants::zero(), TinyConstants::zero()));
    world.step(dt);
  }

  // Compute error
  TinyVector3 delta = bodies[TARGET_ID]->m_world_pose.m_position - goal;
  return delta.sqnorm();
}

// // Computes gradient using finite differences
// void grad_finite(double force_x, double force_y, double* cost,
//                  double* d_force_x, double* d_force_y, int steps = 300,
//                  double eps = 1e-5) {
//   *cost = rollout<double, DoubleUtils>(force_x, force_y, steps);
//   double cx = rollout<double, DoubleUtils>(force_x + eps, force_y, steps);
//   double cy = rollout<double, DoubleUtils>(force_x, force_y + eps, steps);
//   *d_force_x = (cx - *cost) / eps;
//   *d_force_y = (cy - *cost) / eps;
// }

// void grad_stan(double force_x, double force_y, double* cost, double*
// d_force_x,
//               double* d_force_y, int steps = 300, double eps = 1e-5) {
//  standouble fx = force_x;
//  fx.d_ = 1;
//  standouble fy = force_y;
//
//  standouble c = rollout<standouble, StanDoubleUtils>(fx, fy, steps);
//  *cost = c.val();
//  *d_force_x = c.tangent();
//
//  fx.d_ = 0;
//  fy.d_ = 1;
//  c = rollout<standouble, StanDoubleUtils>(fx, fy, steps);
//  *d_force_y = c.tangent();
//}

// void grad_dual(double force_x, double force_y, double* cost, double* d_force_x,
//                double* d_force_y, int steps = 300, double eps = 1e-5) {
//   typedef TinyDual<double> TinyDual;
//   {
//     TinyDual fx(force_x, 1.);
//     TinyDual fy(force_y, 0.);

//     TinyDual c = rollout<TinyDual, TinyDualDoubleUtils>(fx, fy, steps);
//     *cost = c.real();
//     *d_force_x = c.dual();
//   }
//   {
//     TinyDual fx(force_x, 0.);
//     TinyDual fy(force_y, 1.);

//     TinyDual c = rollout<TinyDual, TinyDualDoubleUtils>(fx, fy, steps);
//     *d_force_y = c.dual();
//   }
// }

struct CeresFunctional {
  int steps{20};

  template <typename T>
  bool operator()(const T* const x, T* e) const {
    // typedef ceres::Jet<double, 2> Jet;
    // T fx(x); //TODO convert array x to vector fx
    std::vector<T>fx(x, x+steps); //vector from input array
    typedef std::conditional_t<std::is_same_v<T, double>, DoubleUtils,
                               CeresUtils<2>>
        Utils;
    *e = rollout<T, Utils>(fx, steps);
    return true;
  }
};

ceres::AutoDiffCostFunction<CeresFunctional, 1, 2> cost_function(
    new CeresFunctional);
const int steps{20}; //TODO merge with CeresFunctional above
double* parameters = new double[steps];
double* gradient = new double[steps];

void grad_ceres(double* force_x, double* cost, double* d_force_x, int steps = steps) {
  // parameters[0] = force_x;
  // parameters[1] = force_y;
  double const* const* params = &force_x;
  cost_function.Evaluate(params, cost, &gradient);
  *d_force_x = gradient;
}

int main(int argc, char* argv[]) {
  TinyFileUtils::find_file("sphere2red.urdf", sphere2red);
  std::string connection_mode = "shared_memory"; //shared memory requires pybullet server to be running in the background

  using namespace std::chrono;

  visualizer = new VisualizerAPI;
  visualizer->setTimeOut(1e20);
  printf("mode=%s\n", const_cast<char*>(connection_mode.c_str()));
  int mode = eCONNECT_GUI;
  if (connection_mode == "direct") mode = eCONNECT_DIRECT;
  if (connection_mode == "shared_memory") mode = eCONNECT_SHARED_MEMORY;
  visualizer->connect(mode);

  visualizer->resetSimulation();

  double init_force_x = 0.;
  int steps = 20; //TODO: merge with other instantiations of step
  double cost;
  double learning_rate = 1e2;
  std::vector<double> force_x(steps, init_force_x);
  std::vector<double> d_force_x(steps, 0.0);
  for (int iter = 0; iter < gd_steps; ++iter) {
    grad_ceres(force_x, &cost, &d_force_x, steps);
    printf("Iteration %02d - cost: %.3f \tforce: [%.2f]\n", iter, cost,
            force_x);
    force_x -= learning_rate * d_force_x;
  }
  rollout<double, DoubleUtils>(force_x, steps);

  visualizer->disconnect();
  delete visualizer;

  return 0;
}
