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


#include <chrono>
#include <iostream>
#include <thread>

#define BULLET_VISUALIZER true


#if !BULLET_VISUALIZER
#include "opengl_window/tiny_opengl3_app.h"
#endif

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
  // tds::FileUtils::find_file("sphere8cube.urdf", urdf_filename);
  // bool floating_base = true;
  tds::FileUtils::find_file("pendulum5.urdf", urdf_filename);
  bool floating_base = false;
  std::string plane_filename;
  tds::FileUtils::find_file("plane_implicit.urdf", plane_filename);

  if (argc > 1) urdf_filename = std::string(argv[1]);

  tds::activate_nan_trap();

#if BULLET_VISUALIZER
  printf("floating_base=%d\n", floating_base);
  printf("urdf_filename=%s\n", urdf_filename.c_str());
  VisualizerAPI *sim2 = new VisualizerAPI();
  bool isConnected2 = sim2->connect(eCONNECT_DIRECT);

  VisualizerAPI *sim = new VisualizerAPI();

  printf("connection_mode=%s\n", connection_mode.c_str());
  int mode = eCONNECT_DIRECT;
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
  int grav_id = sim->addUserDebugParameter("gravity", -10, 10, -2.);
  #endif

  // int rotateCamera = 0;

  tds::World<Algebra> world;
  MultiBody *system = nullptr;

  if (true) {
  system = world.create_multi_body();
  tds::SystemConstructor constructor(urdf_filename, plane_filename);
  constructor.is_floating = floating_base;
  VisualizerAPI *sim_ = new VisualizerAPI();
  sim_->connect(eCONNECT_DIRECT);
  #if BULLET_VISUALIZER
  constructor(sim2, sim, world, &system);
  #else
  constructor(sim_, sim_, world, &system);
  #endif
  } else {
  tds::UrdfCache<Algebra> cache;
  system = cache.construct(urdf_filename, world, false, floating_base);
  }

#if !BULLET_VISUALIZER
  TinyOpenGL3App app("generic_urdf_collision", 1024, 768);
  app.m_renderer->init();
  app.set_up_axis(2);
  app.m_renderer->get_active_camera()->set_camera_distance(4);
  app.m_renderer->get_active_camera()->set_camera_pitch(-30);
  app.m_renderer->get_active_camera()->set_camera_target_position(0, 0, 0);
  std::vector<int> tiny_visual_ids;
  for (std::size_t i = 0; i < system->size(); ++i) {
    int cube_shape = app.register_cube_shape(0.05f, 0.05f, 0.05f);
    int cube_id = app.m_renderer->register_graphics_instance(cube_shape);
    tiny_visual_ids.push_back(cube_id);
  }
  #endif

  world.set_mb_constraint_solver(
      new tds::MultiBodyConstraintSolverSpring<Algebra>);

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
    // system->qd()[1] = 2;
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
  } else {
    system->base_X_world().translation = Algebra::unit3_z();
  }
  system->print_state();

  double dt = 1. / 1000.;
  double time = 0;
  int step = 0;
  while (true) {
    #if BULLET_VISUALIZER
    double gravZ = sim->readUserDebugParameter(grav_id);
    #else
    double gravZ = -9.81;
    #endif
    world.set_gravity(Vector3(0, 0, gravZ));

    {
      // sim->submitProfileTiming("forwardDynamics");
      forward_dynamics(*system, world.get_gravity());
      // sim->submitProfileTiming("");
      system->clear_forces();
    }

#if BULLET_VISUALIZER
      tds::PyBulletUrdfImport<Algebra>::sync_graphics_transforms(system, *sim);
    #else
    if (step % 10 == 0) {
      auto& mb = *system;
      TinyVector3f parent_pos(
        static_cast<float>(mb.base_X_world().translation[0]),
        static_cast<float>(mb.base_X_world().translation[1]),
        static_cast<float>(mb.base_X_world().translation[2]));
    for (const auto &link : mb) {
    app.m_renderer->update_camera(2);
    DrawGridData data;
    data.upAxis = 2;
    app.draw_grid(data);

      TinyVector3f link_pos(static_cast<float>(link.X_world.translation[0]),
                            static_cast<float>(link.X_world.translation[1]),
                            static_cast<float>(link.X_world.translation[2]));

      app.m_renderer->draw_line(link_pos, parent_pos,
                                TinyVector3f(0.5, 0.5, 0.5), 2.f);
      parent_pos = link_pos;
      std::size_t j = 0;
        tds::Transform<Algebra> X_visual = link.X_world * link.X_visuals[j];
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
            geom_pos, geom_orn, tiny_visual_ids[link.index]);
        TinyVector3f color(0.1, 0.6, 0.8);
        app.m_renderer->draw_line(link_pos, geom_pos, color, 2.f);
    }
    app.m_renderer->render_scene();
    app.m_renderer->write_transforms();
    app.swap_buffer();
    }
    #endif
    std::this_thread::sleep_for(std::chrono::duration<double>(dt/5.));

    {
      // sim->submitProfileTiming("world_step");
      world.step(dt);
      // sim->submitProfileTiming("");
      time += dt;
    }

    {
      // sim->submitProfileTiming("integrate");
      integrate_euler(*system, dt);
      system->print_state();
      // sim->submitProfileTiming("");
    }
    // sim->setGravity(btVector3(0, 0, gravZ));
    ++step;
  }

  // sim->disconnect();
  // sim2->disconnect();

  // delete sim;
  // delete sim2;

  return EXIT_SUCCESS;
}
