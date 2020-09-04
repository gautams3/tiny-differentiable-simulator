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
const int TARGET_ID = 5;

using namespace tds;

template <typename Algebra>
typename Algebra::Scalar rollout(
    typename Algebra::Scalar force_x, typename Algebra::Scalar force_y,
    int steps = 300, VisualizerAPI* vis = nullptr,
    typename Algebra::Scalar dt = Algebra::fraction(1, 60)) {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  typedef tds::RigidBody<Algebra> RigidBody;
  typedef tds::Geometry<Algebra> Geometry;

  printf("main.cpp:%i\n", __LINE__);
  fflush(stdout);

  std::vector<int> visuals;
  Vector3 target(Algebra::fraction(35, 10), Algebra::fraction(8, 1),
                 Algebra::zero());
  printf("main.cpp:%i\n", __LINE__);
  fflush(stdout);
  if (vis) {
    vis->resetSimulation();
  }

  Scalar gravity_z = Algebra::zero();
  tds::World<Algebra> world(gravity_z);
  printf("main.cpp:%i\n", __LINE__);
  fflush(stdout);

  std::vector<RigidBody*> bodies;

  Scalar radius = Algebra::half();
  Scalar mass = Algebra::one();
  Scalar deg_60 = Algebra::pi() / Algebra::fraction(3, 1);  // even triangle
  Scalar dx = Algebra::cos(deg_60) * radius * Algebra::two();
  Scalar dy = Algebra::sin(deg_60) * radius * Algebra::two();
  Scalar rx = Algebra::zero(), y = Algebra::zero();
  printf("main.cpp:%i\n", __LINE__);
  fflush(stdout);
  int ball_id = 0;
  for (int column = 1; column <= 3; ++column) {
    Scalar x = rx;
    for (int i = 0; i < column; ++i) {
      const Geometry* geom = world.create_sphere(radius);
  printf("main.cpp:%i\n", __LINE__);
  fflush(stdout);
      RigidBody* body = world.create_rigid_body(mass, geom);
  printf("main.cpp:%i\n", __LINE__);
  fflush(stdout);
      body->world_pose().position = Vector3(x, y, Algebra::zero());
  printf("main.cpp:%i\n", __LINE__);
  fflush(stdout);
      bodies.push_back(body);
      if (vis) {
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
      }
      ++ball_id;
      x += radius * Algebra::two();
    }
    rx = rx - dx;
    y = y + dy;
  }
  printf("main.cpp:%i\n", __LINE__);
  fflush(stdout);

  // Create white ball
  Vector3 white = Vector3(Algebra::zero(), -Algebra::two(), Algebra::zero());
  const Geometry* white_geom = world.create_sphere(radius);
  RigidBody* white_ball = world.create_rigid_body(mass, white_geom);
  white_ball->world_pose().position = Vector3(white.x(), white.y(), white.z());
  bodies.push_back(white_ball);
  white_ball->apply_central_force(Vector3(force_x, force_y, Algebra::zero()));

  printf("main.cpp:%i\n", __LINE__);
  fflush(stdout);
  if (vis) {
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
  }

  for (int i = 0; i < steps; i++) {
  printf("main.cpp:%i\n", __LINE__);
  fflush(stdout);
    visualizer->submitProfileTiming("world.step");
    world.step(dt);
    visualizer->submitProfileTiming("");
  printf("main.cpp:%i\n", __LINE__);
  fflush(stdout);

    if (vis) {
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

  // Compute error
  Vector3 delta = bodies[TARGET_ID]->world_pose().position - target;
  return Algebra::sqnorm(delta);
}

// Computes gradient using finite differences
void grad_finite(double force_x, double force_y, double* cost,
                 double* d_force_x, double* d_force_y, int steps = 300,
                 double eps = 1e-5) {
  using Algebra = TinyAlgebra<double, DoubleUtils>;
  *cost = rollout<Algebra>(force_x, force_y, steps);
  double cx = rollout<Algebra>(force_x + eps, force_y, steps);
  double cy = rollout<Algebra>(force_x, force_y + eps, steps);
  *d_force_x = (cx - *cost) / eps;
  *d_force_y = (cy - *cost) / eps;
}

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

void grad_dual(double force_x, double force_y, double* cost, double* d_force_x,
               double* d_force_y, int steps = 300, double eps = 1e-5) {
  typedef TinyDual<double> Dual;
  using Algebra = TinyAlgebra<Dual, TinyDualDoubleUtils>;
  {
    Dual fx(force_x, 1.);
    Dual fy(force_y, 0.);

    Dual c = rollout<Algebra>(fx, fy, steps);
    *cost = c.real();
    *d_force_x = c.dual();
  }
  {
    Dual fx(force_x, 0.);
    Dual fy(force_y, 1.);

    Dual c = rollout<Algebra>(fx, fy, steps);
    *d_force_y = c.dual();
  }
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
    *e = rollout<TinyAlgebra<T, Utils>>(fx, fy, steps);
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

void grad_enoki(double force_x, double force_y, double* cost, double* d_force_x,
                double* d_force_y, int steps = 300) {
                  using FloatC    = enoki::CUDAArray<float>;
  using FloatD = enoki::DiffArray<FloatC>;
  enoki::Matrix<float, 3> matf(1.);
  std::cout << matf << std::endl;
  enoki::Matrix<FloatD, 3> mat(1.);
  std::cout << mat << std::endl;
  enoki::requires_gradient(mat(0,0));
  enoki::Matrix<FloatD, 3> mat2(2.);
  std::cout << mat2 << std::endl;
enoki::Matrix<FloatD, 3> result = mat * mat2;
  std::cout << result << std::endl;
  enoki::backward(mat(0,0));

  std::cout << "Gradient: " << enoki::gradient(mat(0,0)) << std::endl;




  // using Algebra = EnokiAlgebraT<FloatD>;
  // FloatD fx(force_x), fy(force_y);
  // enoki::set_requires_gradient(fx);
  // enoki::set_requires_gradient(fy);

  // typename Algebra::Matrix3 mat = Algebra::eye3();
  // typename Algebra::Vector3 vec = Algebra::zero3();

  // typename Algebra::Vector3  temp = mat * vec;
  // Algebra::print("mat * vec", temp);

  // FloatD e = rollout<Algebra>(fx, fy, steps);
  // printf("main.cpp:%i\n", __LINE__);
  // fflush(stdout);
  // enoki::backward(e);
  // printf("main.cpp:%i\n", __LINE__);
  // fflush(stdout);
  // *cost = Algebra::to_double(e);
  // printf("main.cpp:%i\n", __LINE__);
  // fflush(stdout);
  // *d_force_x = enoki::gradient(fx);
  // printf("main.cpp:%i\n", __LINE__);
  // fflush(stdout);
  // *d_force_y = enoki::gradient(fy);
  // printf("main.cpp:%i\n", __LINE__);
  // fflush(stdout);
}

int main(int argc, char* argv[]) {
  tds::FileUtils::find_file("sphere2red.urdf", sphere2red);
  std::string connection_mode = "gui";

  using namespace std::chrono;

  visualizer = new VisualizerAPI;
  visualizer->setTimeOut(1e30);
  printf("mode=%s\n", const_cast<char*>(connection_mode.c_str()));
  int mode = eCONNECT_GUI;
  if (connection_mode == "direct") mode = eCONNECT_DIRECT;
  if (connection_mode == "shared_memory") mode = eCONNECT_SHARED_MEMORY;
  visualizer->connect(mode);

  visualizer->resetSimulation();

  double init_force_x = 0., init_force_y = 500.;
  int steps = 300;
  // using DoubleAlgebra = TinyAlgebra<double, DoubleUtils>;
  // rollout<DoubleAlgebra>(init_force_x, init_force_y, steps, visualizer);

  // {
  //   auto start = high_resolution_clock::now();
  //   double cost, d_force_x, d_force_y;
  //   double learning_rate = 1e2;
  //   double force_x = init_force_x, force_y = init_force_y;
  //   for (int iter = 0; iter < 50; ++iter) {
  //     grad_finite(force_x, force_y, &cost, &d_force_x, &d_force_y, steps);
  //     printf("Iteration %02d - cost: %.3f \tforce: [%.2f %2.f]\n", iter, cost,
  //            force_x, force_y);
  //     force_x -= learning_rate * d_force_x;
  //     force_y -= learning_rate * d_force_y;
  //   }
  //   auto stop = high_resolution_clock::now();
  //   auto duration = duration_cast<microseconds>(stop - start);
  //   printf("Finite differences took %ld microseconds.",
  //          static_cast<long>(duration.count()));
  //   fflush(stdout);
  // }
  // //  {
  // //    auto start = high_resolution_clock::now();
  // //    double cost, d_force_x, d_force_y;
  // //    double learning_rate = 1e2;
  // //    double force_x = init_force_x, force_y = init_force_y;
  // //    for (int iter = 0; iter < 50; ++iter) {
  // //      grad_stan(force_x, force_y, &cost, &d_force_x, &d_force_y, steps);
  // //      printf("Iteration %02d - cost: %.3f \tforce: [%.2f %2.f]\n", iter,
  // //      cost,
  // //             force_x, force_y);
  // //      force_x -= learning_rate * d_force_x;
  // //      force_y -= learning_rate * d_force_y;
  // //    }
  // //    auto stop = high_resolution_clock::now();
  // //    auto duration = duration_cast<microseconds>(stop - start);
  // //    printf("Stan's forward-mode AD took %ld microseconds.",
  // //    static_cast<long>(duration.count()));
  // fflush(stdout);
  // //  }
  // {
  //   auto start = high_resolution_clock::now();
  //   double cost, d_force_x, d_force_y;
  //   double learning_rate = 1e2;
  //   double force_x = init_force_x, force_y = init_force_y;
  //   for (int iter = 0; iter < 50; ++iter) {
  //     grad_dual(force_x, force_y, &cost, &d_force_x, &d_force_y, steps);
  //     printf("Iteration %02d - cost: %.3f \tforce: [%.2f %2.f]\n", iter, cost,
  //            force_x, force_y);
  //     force_x -= learning_rate * d_force_x;
  //     force_y -= learning_rate * d_force_y;
  //   }
  //   auto stop = high_resolution_clock::now();
  //   auto duration = duration_cast<microseconds>(stop - start);
  //   printf("TinyDual took %ld microseconds.",
  //          static_cast<long>(duration.count()));
  //   fflush(stdout);
  // }
  // {
  //   auto start = high_resolution_clock::now();
  //   double cost, d_force_x, d_force_y;
  //   double learning_rate = 1e2;
  //   double force_x = init_force_x, force_y = init_force_y;
  //   for (int iter = 0; iter < 50; ++iter) {
  //     grad_ceres(force_x, force_y, &cost, &d_force_x, &d_force_y, steps);
  //     printf("Iteration %02d - cost: %.3f \tforce: [%.2f %2.f]\n", iter, cost,
  //            force_x, force_y);
  //     force_x -= learning_rate * d_force_x;
  //     force_y -= learning_rate * d_force_y;
  //   }
  //   auto stop = high_resolution_clock::now();
  //   auto duration = duration_cast<microseconds>(stop - start);
  //   printf("Ceres Jet took %ld microseconds.",
  //          static_cast<long>(duration.count()));
  //   fflush(stdout);
  //   // rollout<DoubleAlgebra>(force_x, force_y, steps, visualizer);
  // }
  {
    auto start = high_resolution_clock::now();
    double cost, d_force_x, d_force_y;
    double learning_rate = 1e2;
    double force_x = init_force_x, force_y = init_force_y;
    for (int iter = 0; iter < 50; ++iter) {
      grad_enoki(force_x, force_y, &cost, &d_force_x, &d_force_y, steps);
      printf("Iteration %02d - cost: %.3f \tforce: [%.2f %2.f]\n", iter, cost,
             force_x, force_y);
      force_x -= learning_rate * d_force_x;
      force_y -= learning_rate * d_force_y;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("Enoki took %ld microseconds.", static_cast<long>(duration.count()));
    fflush(stdout);
    // rollout<DoubleAlgebra>(force_x, force_y, steps, visualizer);
  }

  visualizer->disconnect();
  delete visualizer;

  return 0;
}
