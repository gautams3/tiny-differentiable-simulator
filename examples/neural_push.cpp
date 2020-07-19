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
#include <highfive/H5Easy.hpp>
#include <iostream>
#include <thread>

#include "Utils/b3Clock.h"
#include "pybullet_visualizer_api.h"
#include "tiny_double_utils.h"
#include "tiny_file_utils.h"
#include "tiny_mb_constraint_solver_spring.h"
#include "tiny_multi_body.h"
#include "tiny_system_constructor.h"

typedef PyBulletVisualizerAPI VisualizerAPI;

struct PushData {
  std::vector<double> tip_x, tip_y, tip_yaw;
  std::vector<double> time;
  std::vector<double> object_x, object_y, object_yaw;
  std::vector<double> tip_wrench_x, tip_wrench_y, tip_wrench_yaw;

  PushData(const std::string &filename) {
    std::vector<std::vector<double>> vecs;
    H5Easy::File push_file(filename, H5Easy::File::ReadOnly);
    HighFive::DataSet data = push_file.getDataSet("tip_pose");
    data.read(vecs);
    for (std::size_t i = 0; i < vecs.size(); ++i) {
      const auto &vec = vecs[i];
      time.push_back(vec[0]);  // times are very similar for all datasets
      tip_x.push_back(vec[1]);
      tip_y.push_back(vec[2]);
      tip_yaw.push_back(vec[3]);
    }
    vecs.clear();

    data = push_file.getDataSet("object_pose");
    data.read(vecs);
    for (std::size_t i = 0; i < vecs.size(); ++i) {
      const auto &vec = vecs[i];
      object_x.push_back(vec[1]);
      object_y.push_back(vec[2]);
      object_yaw.push_back(vec[3]);
    }
    vecs.clear();

    data = push_file.getDataSet("ft_wrench");
    data.read(vecs);
    for (std::size_t i = 0; i < vecs.size(); ++i) {
      const auto &vec = vecs[i];
      tip_wrench_x.push_back(vec[1]);
      tip_wrench_y.push_back(vec[2]);
      tip_wrench_yaw.push_back(vec[3]);
    }
    vecs.clear();

    // trim the datasets to match in length
    std::size_t min_len =
        std::min({time.size(), object_x.size(), tip_wrench_x.size()});
    time.resize(min_len);
    tip_x.resize(min_len);
    tip_y.resize(min_len);
    tip_yaw.resize(min_len);
    object_x.resize(min_len);
    object_y.resize(min_len);
    object_yaw.resize(min_len);
    tip_wrench_x.resize(min_len);
    tip_wrench_y.resize(min_len);
    tip_wrench_yaw.resize(min_len);

    assert(tip_x.size() == tip_y.size() && tip_y.size() == tip_yaw.size() &&
           tip_yaw.size() == time.size() && time.size() == object_x.size() &&
           object_x.size() == object_y.size() &&
           object_y.size() == object_yaw.size() &&
           object_yaw.size() == tip_wrench_x.size() &&
           tip_wrench_x.size() == tip_wrench_y.size() &&
           tip_wrench_y.size() == tip_wrench_yaw.size());

    std::cout << "Read push dataset \"" + filename + "\" with " << tip_x.size()
              << " entries.\n";
  }
};

int main(int argc, char *argv[]) {
  std::string connection_mode = "gui";

  std::string object_filename;
  TinyFileUtils::find_file("mit-push/obj/rect1.urdf", object_filename);
  std::string tip_filename;
  TinyFileUtils::find_file("mit-push/obj/tip.urdf", tip_filename);
  std::string plane_filename;
  TinyFileUtils::find_file("mit-push/obj/plywood.urdf", plane_filename);

  std::string push_filename;
  TinyFileUtils::find_file(
      "mit-push/abs_rect1/"
      "motion_surface=abs_shape=rect1_a=0_v=20_i=0.000_s=0.700_t=0.000_rep="
      "0004.h5",
      push_filename);

  PushData data(push_filename);

  if (argc > 1) object_filename = std::string(argv[1]);
  bool floating_base = true;

  // Set NaN trap
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  printf("floating_base=%d\n", floating_base);
  printf("object_filename=%s\n", object_filename.c_str());
  printf("tip_filename=%s\n", tip_filename.c_str());
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

  typedef double Scalar;
  typedef DoubleUtils Utils;

  TinyUrdfCache<Scalar, Utils> urdf_cache;

  TinyWorld<Scalar, Utils> world;
  TinyMultiBody<Scalar, Utils> *object = world.create_multi_body();
  TinyMultiBody<Scalar, Utils> *tip = world.create_multi_body();
  TinySystemConstructor<> constructor(object_filename, plane_filename);
  constructor(sim2, sim, world, &object);

  auto tip_urdf =
      urdf_cache.template retrieve<VisualizerAPI>(tip_filename, sim2, sim);
  TinyUrdfToMultiBody<Scalar, Utils>::convert_to_multi_body(tip_urdf, world,
                                                            *tip);
  tip->initialize();

  // delete world.m_mb_constraint_solver;
  // world.m_mb_constraint_solver =
  //     new TinyMultiBodyConstraintSolverSpring<Scalar, Utils>;

  // object->m_q[2] = 0.5;
  fflush(stdout);

  object->print_state();

  double dt = 1. / 1000.;
  double time = 0;

  while (true) {
    printf("Playback...\n");
    for (std::size_t i = 0; i < data.time.size(); ++i) {
      object->m_q[0] = data.object_x[i];
      object->m_q[1] = data.object_y[i];
      object->m_q[3] = data.object_yaw[i];
      object->forward_kinematics();
      PyBulletUrdfImport<Scalar, Utils>::sync_graphics_transforms(object, *sim);
      tip->m_q[0] = data.tip_x[i];
      tip->m_q[1] = data.tip_y[i];
      tip->forward_kinematics();
      PyBulletUrdfImport<Scalar, Utils>::sync_graphics_transforms(tip, *sim);
      std::this_thread::sleep_for(std::chrono::duration<double>(dt));
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(5.));
  }

  // while (sim->canSubmitCommand()) {
  //   double gravZ = sim->readUserDebugParameter(grav_id);
  //   world.set_gravity(TinyVector3<Scalar, Utils>(0, 0, gravZ));

  //   {
  //     // object->control(dt, control);
  //     sim->submitProfileTiming("forwardDynamics");
  //     object->forward_dynamics(world.get_gravity());
  //     sim->submitProfileTiming("");
  //     PyBulletUrdfImport<Scalar, Utils>::sync_graphics_transforms(object,
  //     *sim); object->clear_forces();
  //   }

  //   {
  //     sim->submitProfileTiming("integrate_q");
  //     // object->integrate_q(dt);  //??
  //     sim->submitProfileTiming("");
  //   }

  //   {
  //     sim->submitProfileTiming("world_step");
  //     world.step(dt);
  //     fflush(stdout);
  //     sim->submitProfileTiming("");
  //     time += dt;
  //   }

  //   {
  //     sim->submitProfileTiming("integrate");
  //     object->integrate(dt);
  //     // object->print_state();
  //     sim->submitProfileTiming("");
  //   }
  //   std::this_thread::sleep_for(std::chrono::duration<double>(dt));
  //   sim->setGravity(btVector3(0, 0, gravZ));
  // }

  sim->disconnect();
  sim2->disconnect();

  // delete sim;
  // delete sim2;

  return EXIT_SUCCESS;
}
