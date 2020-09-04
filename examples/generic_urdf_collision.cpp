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

#include <fenv.h>
#include <stdio.h>

#include <chrono>
#include <iostream>
#include <thread>

#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "math/tiny/tiny_algebra.hpp"
#include "math/tiny/tiny_double_utils.h"
#include "mb_constraint_solver_spring.hpp"
#include "multi_body.hpp"
#include "pybullet_visualizer_api.h"
#include "urdf/pybullet_urdf_import.hpp"
#include "urdf/system_constructor.hpp"
#include "urdf/urdf_cache.hpp"
#include "utils/file_utils.hpp"
#include "world.hpp"

typedef PyBulletVisualizerAPI VisualizerAPI;

int main(int argc, char *argv[]) {
  std::string connection_mode = "gui";

  typedef TinyAlgebra<double, DoubleUtils> Algebra;
  typedef typename Algebra::Vector3 Vector3;
  typedef typename Algebra::Quaternion Quaternion;
  typedef typename Algebra::VectorX VectorX;
  typedef typename Algebra::Matrix3 Matrix3;
  typedef typename Algebra::Matrix3X Matrix3X;
  typedef tds::MultiBody<Algebra> MultiBody;
  typedef tds::MultiBodyContactPoint<Algebra> ContactPoint;

  std::string urdf_filename;
  //"cheetah_link0_1.urdf"
  //"pendulum5.urdf"
  //"sphere2.urdf"
  tds::FileUtils::find_file("sphere8cube.urdf", urdf_filename);
  std::string plane_filename;
  tds::FileUtils::find_file("plane_implicit.urdf", plane_filename);

  if (argc > 1) urdf_filename = std::string(argv[1]);
  bool floating_base = true;

  // Set NaN trap
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  printf("floating_base=%d\n", floating_base);
  printf("urdf_filename=%s\n", urdf_filename.c_str());
  VisualizerAPI *sim2 = new VisualizerAPI();
  bool isConnected2 = sim2->connect(eCONNECT_DIRECT);

  VisualizerAPI *sim = new VisualizerAPI();

  printf("connection_mode=%s\n", connection_mode.c_str());
  int mode = eCONNECT_GUI;
  if (connection_mode == "direct") mode = eCONNECT_DIRECT;
  if (connection_mode == "gui") mode = eCONNECT_GUI;
  if (connection_mode == "shared_memory") mode = eCONNECT_SHARED_MEMORY;

  bool isConnected = sim->connect(mode);
  if (!isConnected) {
    printf("Cannot connect\n");
    return -1;
  }

  sim->resetSimulation();
  sim->setTimeOut(10);
  int grav_id = sim->addUserDebugParameter("gravity", -10, 10, -2);

  int rotateCamera = 0;

  tds::World<Algebra> world;
  MultiBody *system = world.create_multi_body();
  tds::SystemConstructor constructor(urdf_filename, plane_filename);
  constructor.is_floating = floating_base;
  constructor(sim2, sim, world, &system);
  
  world.set_mb_constraint_solver(
      new tds::MultiBodyConstraintSolverSpring<Algebra>);

  //  system->q()[0] = 2.;
  //  system->q()[1] = 1.2;
  //  system->q()[2] = 0.1;
  //  system->qd()[3] = 5;
  system->q()[1] = .2;
  fflush(stdout);

  if (floating_base) {
    Quaternion start_rot;
    start_rot.set_euler_rpy(Vector3(0.8, 1.1, 0.9));
    const double initial_height = 1.2;
    const Vector3 initial_velocity(0.7, 2., 0.);
    system->q()[0] = start_rot.x();
    system->q()[1] = start_rot.y();
    system->q()[2] = start_rot.z();
    system->q()[3] = start_rot.w();
    system->q()[4] = 0.;
    system->q()[5] = 0.;
    system->q()[6] = initial_height;

    system->qd()[0] = 0.;
    system->qd()[1] = 0.;
    system->qd()[2] = 0.;
    system->qd()[3] = initial_velocity.x();
    system->qd()[4] = initial_velocity.y();
    system->qd()[5] = initial_velocity.z();

    // apply some "random" rotation
    // system->q()[0] = 0.06603363263475902;
    // system->q()[1] = 0.2764891273883223;
    // system->q()[2] = 0.2477976811032405;
    // system->q()[3] = 0.9261693317298725;
    // system->q()[6] = 2;
  }
  system->print_state();

  double dt = 1. / 1000.;
  double time = 0;
  while (sim->canSubmitCommand()) {
    double gravZ = sim->readUserDebugParameter(grav_id);
    world.set_gravity(Vector3(0, 0, gravZ));

    {
      // system->control(dt, control);
      sim->submitProfileTiming("forwardDynamics");
      forward_dynamics(*system, world.get_gravity());
      sim->submitProfileTiming("");
      tds::PyBulletUrdfImport<Algebra>::sync_graphics_transforms(system, *sim);
      system->clear_forces();
    }

    {
      sim->submitProfileTiming("integrate_q");
      // system->integrate_q(dt);  //??
      sim->submitProfileTiming("");
    }

    {
      sim->submitProfileTiming("world_step");
      world.step(dt);
      fflush(stdout);
      sim->submitProfileTiming("");
      time += dt;
    }

    {
      sim->submitProfileTiming("integrate");
      integrate_euler(*system, dt);
      // system->print_state();
      sim->submitProfileTiming("");
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(dt));
    sim->setGravity(btVector3(0, 0, gravZ));
  }

  sim->disconnect();
  sim2->disconnect();

  // delete sim;
  // delete sim2;

  return EXIT_SUCCESS;
}
