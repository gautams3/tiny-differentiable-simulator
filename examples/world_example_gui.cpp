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
#include <chrono>  // std::chrono::seconds
#include <thread>  // std::this_thread::sleep_for

#include "math/pose.hpp"
#include "math/tiny/tiny_double_utils.h"
#include "multi_body.hpp"
#include "pybullet_visualizer_api.h"
#include "rigid_body.hpp"
#include "utils/file_utils.hpp"
#include "world.hpp"

typedef PyBulletVisualizerAPI VisualizerAPI;
typedef TinyAlgebra<double, DoubleUtils> Algebra;
typedef typename Algebra::Vector3 Vector3;

int main(int argc, char* argv[]) {
  std::string connection_mode = "gui";
  auto* visualizer = new VisualizerAPI;
  visualizer->setTimeOut(1e30);
  printf("\nmode=%s\n", const_cast<char*>(connection_mode.c_str()));
  int mode = eCONNECT_GUI;
  if (connection_mode == "direct") mode = eCONNECT_DIRECT;
  if (connection_mode == "shared_memory") {
    mode = eCONNECT_SHARED_MEMORY;
    printf(
        "Shared memory mode: Ensure you have visualizer server running (e.g. "
        "python -m pybullet_utils.runServer)\n");
  }
  visualizer->connect(mode);

  visualizer->resetSimulation();

  tds::World<Algebra> world;

  {
    double mass = 0.0;
    std::string filename;
    tds::FileUtils::find_file("plane_implicit.urdf", filename);
    visualizer->loadURDF(filename);
    const tds::Geometry<Algebra>* geom = world.create_plane();
    tds::RigidBody<Algebra>* body = world.create_rigid_body(mass, geom);
  }
  std::vector<tds::RigidBody<Algebra>*> bodies;
  std::vector<int> visuals;

  // MultiBody<Algebra>* mb = world.create_multi_body();
  // init_xarm6<Algebra>(*mb);

  {
    std::string filename;
    tds::FileUtils::find_file("sphere2.urdf", filename);
    int sphereId = visualizer->loadURDF(filename);
    double mass = 0.0;
    double radius = 0.5;
    const tds::Geometry<Algebra>* geom = world.create_sphere(radius);
    tds::RigidBody<Algebra>* body = world.create_rigid_body(mass, geom);
    body->world_pose().position = Vector3(0, 0, 0.5);
    bodies.push_back(body);
    visuals.push_back(sphereId);
  }
  {
    std::string filename;
    tds::FileUtils::find_file("sphere2.urdf", filename);
    int sphereId = visualizer->loadURDF(filename);
    double mass = 1.0;

    double radius = 0.5;
    const tds::Geometry<Algebra>* geom = world.create_sphere(radius);
    tds::RigidBody<Algebra>* body = world.create_rigid_body(mass, geom);
    body->world_pose().position = Vector3(0, 0.22, 1.5);
    bodies.push_back(body);
    visuals.push_back(sphereId);
  }

  // std::vector<double> q;
  // std::vector<double> qd;
  // std::vector<double> tau;
  // std::vector<double> qdd;
  // Vector3<Algebra> gravity(0., 0., -9.81);

  // for (int i = 0; i < mb->m_links.size(); i++) {
  //   q.push_back(DoubleUtils::fraction(1, 10) +
  //               DoubleUtils::fraction(1, 10) * DoubleUtils::fraction(i,
  //               1));
  //   qd.push_back(DoubleUtils::fraction(3, 10) +
  //                DoubleUtils::fraction(1, 10) *
  //                    DoubleUtils::fraction(i, 1));
  //   tau.push_back(DoubleUtils::zero());
  //   qdd.push_back(DoubleUtils::zero());
  // }

  double dt = 1. / 60.;
  for (int i = 0; i < 300; i++) {
    {
      visualizer->submitProfileTiming("world.step");
      world.step(dt);
      visualizer->submitProfileTiming("");
    }
    //   {
    //       // visualizer->submitProfileTiming("xarm_fk");
    //       // mb->forwardKinematics(q, qd);
    //       // visualizer->submitProfileTiming("");
    //   }

    //   {
    //     visualizer->submitProfileTiming("xarm_aba");
    //     mb->forward_dynamics(q, qd, tau, gravity, qdd);
    //     visualizer->submitProfileTiming("");
    //   }
    //   {
    //     visualizer->submitProfileTiming("xarm_aba");
    //     mb->integrate(q, qd, qdd, dt);
    //     visualizer->submitProfileTiming("");
    //   }
    std::this_thread::sleep_for(std::chrono::duration<double>(dt));
    // sync transforms
    for (std::size_t b = 0; b < bodies.size(); b++) {
      const tds::RigidBody<Algebra>* body = bodies[b];
      int sphereId = visuals[b];
      btVector3 base_pos(body->world_pose().position.getX(),
                         body->world_pose().position.getY(),
                         body->world_pose().position.getZ());
      btQuaternion base_orn(body->world_pose().orientation.getX(),
                            body->world_pose().orientation.getY(),
                            body->world_pose().orientation.getZ(),
                            body->world_pose().orientation.getW());
      visualizer->resetBasePositionAndOrientation(sphereId, base_pos, base_orn);
    }
  }

  return 0;
}
