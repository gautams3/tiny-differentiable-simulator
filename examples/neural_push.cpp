#include <fenv.h>
#include <stdio.h>

#include <chrono>
#include <highfive/H5Easy.hpp>
#include <iostream>
#include <thread>

#include "Utils/b3Clock.h"
#include "pybullet_visualizer_api.h"
#include "tiny_dataset.h"
#include "tiny_double_utils.h"
#include "tiny_file_utils.h"
#include "tiny_mb_constraint_solver_spring.h"
#include "tiny_multi_body.h"
#include "tiny_system_constructor.h"

typedef PyBulletVisualizerAPI VisualizerAPI;

struct PushData {
  std::vector<double> time;
  std::vector<double> tip_x, tip_y, tip_yaw;
  std::vector<double> object_x, object_y, object_yaw;
  std::vector<double> force_x, force_y, force_yaw;

  double dt{0};

  struct Entry {
    double tip_x, tip_y, tip_yaw;
    double object_x, object_y, object_yaw;
    double force_x, force_y, force_yaw;
  };

  PushData(const std::string &filename) {
    std::vector<std::vector<double>> vecs;
    H5Easy::File push_file(filename, H5Easy::File::ReadOnly);
    HighFive::DataSet data = push_file.getDataSet("tip_pose");
    data.read(vecs);
    dt = 0;
    for (std::size_t i = 0; i < vecs.size(); ++i) {
      const auto &vec = vecs[i];
      if (i > 0) {
        // start time from 0
        time.push_back(vec[0] - vecs[0][0]);
        dt += time[i] - time[i - 1];
      } else {
        time.push_back(0);
      }
      tip_x.push_back(vec[1]);
      tip_y.push_back(vec[2]);
      tip_yaw.push_back(vec[3]);
    }
    vecs.clear();

    dt /= time.size() - 1;

    // TODO resample to match tip_pose time steps?
    // (the times are too close to make a difference)

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
      force_x.push_back(vec[1]);
      force_y.push_back(vec[2]);
      force_yaw.push_back(vec[3]);
    }
    vecs.clear();

    // trim the datasets to match in length
    std::size_t min_len =
        std::min({time.size(), object_x.size(), force_x.size()});
    time.resize(min_len);
    tip_x.resize(min_len);
    tip_y.resize(min_len);
    tip_yaw.resize(min_len);
    object_x.resize(min_len);
    object_y.resize(min_len);
    object_yaw.resize(min_len);
    force_x.resize(min_len);
    force_y.resize(min_len);
    force_yaw.resize(min_len);

    assert(tip_x.size() == tip_y.size() && tip_y.size() == tip_yaw.size() &&
           tip_yaw.size() == time.size() && time.size() == object_x.size() &&
           object_x.size() == object_y.size() &&
           object_y.size() == object_yaw.size() &&
           object_yaw.size() == force_x.size() &&
           force_x.size() == force_y.size() &&
           force_y.size() == force_yaw.size());

    std::cout << "Read push dataset \"" + filename + "\" with " << tip_x.size()
              << " entries.\n\tTime step: " << dt << "\n";
  }

  Entry get(double t) const {
    if (t <= 0.0) {
      return Entry{
          tip_x[0],      tip_y[0],   tip_yaw[0], object_x[0],  object_y[0],
          object_yaw[0], force_x[0], force_y[0], force_yaw[0],
      };
    }
    if (t >= time.back()) {
      return Entry{tip_x.back(),    tip_y.back(),    tip_yaw.back(),
                   object_x.back(), object_y.back(), object_yaw.back(),
                   force_x.back(),  force_y.back(),  force_yaw.back()};
    }
    // linear interpolation
    int i = static_cast<int>(std::floor(t / dt + 0.5));
    double alpha = (t - i * dt) / dt;
    return Entry{(1 - alpha) * tip_x[i] + alpha * tip_x[i + 1],
                 (1 - alpha) * tip_y[i] + alpha * tip_y[i + 1],
                 (1 - alpha) * tip_yaw[i] + alpha * tip_yaw[i + 1],
                 (1 - alpha) * object_x[i] + alpha * object_x[i + 1],
                 (1 - alpha) * object_y[i] + alpha * object_y[i + 1],
                 (1 - alpha) * object_yaw[i] + alpha * object_yaw[i + 1],
                 (1 - alpha) * force_x[i] + alpha * force_x[i + 1],
                 (1 - alpha) * force_y[i] + alpha * force_y[i + 1],
                 (1 - alpha) * force_yaw[i] + alpha * force_yaw[i + 1]};
  }
};

// squared distance between points (x1, y1) and (x2, y2)
template <typename Scalar>
Scalar sqr_distance(const Scalar &x1, const Scalar &y1, const Scalar &x2,
                    const Scalar &y2) {
  Scalar dx = x2 - x1, dy = y2 - y1;
  return dx * dx + dy * dy;
}

// Euclidean distance between point (px, py) and line segment (x1, y1, x2, y2)
template <typename Scalar, typename Utils>
Scalar line_distance(const Scalar &px, const Scalar &py, const Scalar &x1,
                     const Scalar &y1, const Scalar &x2, const Scalar &y2,
                     const Scalar &radius, Scalar &line_intersect_x,
                     Scalar &line_intersect_y, Scalar &circle_intersect_x,
                     Scalar &circle_intersect_y) {
  Scalar x21 = x2 - x1;
  Scalar y21 = y2 - y1;

  Scalar norm = x21 * x21 + y21 * y21;

  Scalar u = ((px - x1) * x21 + (py - y1) * y21) / norm;

  if (u > Utils::one())
    u = Utils::one();
  else if (u < Utils::zero())
    u = Utils::zero();

  line_intersect_x = x1 + u * x21;
  line_intersect_y = y1 + u * y21;

  Scalar dx = line_intersect_x - px;
  Scalar dy = line_intersect_y - py;

  Scalar dist = Utils::sqrt1(dx * dx + dy * dy);

  circle_intersect_x = px + radius * dx / dist;
  circle_intersect_y = py + radius * dy / dist;

  return dist;
}

/**
 * Computes a contact point between the tip (circle) and a 2D polygon described
 * by its edges in a contiguous array of points.
 *
 *    Multi body `a` is the tip.
 *    Multi body `b` is the object.
 */
template <typename Scalar, typename Utils>
TinyContactPointMultiBody<Scalar, Utils> compute_contact(
    TinyMultiBody<Scalar, Utils> *tip, TinyMultiBody<Scalar, Utils> *object,
    TinyDataset<Scalar, 2> exterior, const Scalar &tip_radius = 0.0045) {
  const Scalar tip_x = tip->m_q[0];
  const Scalar tip_y = tip->m_q[1];
  // printf("tip:  [%.6f %.6f]\n", tip_x, tip_y);
  TinyContactPointMultiBody<Scalar, Utils> cp;
  cp.m_multi_body_a = tip;
  cp.m_link_a = 1;
  cp.m_multi_body_b = object;
  cp.m_link_b = 3;
  const std::size_t num_ext = exterior.Shape()[0];
  if (num_ext == 0) {
    std::cerr << "Error: Empty exterior passed to compute_contact!\n";
    assert(0);
    return cp;
  }

  // transform exterior points into world frame
  const Scalar &object_x = object->m_q[0];
  const Scalar &object_y = object->m_q[1];
  const Scalar &object_yaw = object->m_q[3];
  const Scalar cos_yaw = Utils::cos1(object_yaw),
               sin_yaw = Utils::sin1(object_yaw);
  Scalar exterior_center_x = Utils::zero();
  Scalar exterior_center_y = Utils::zero();
  for (std::size_t i = 0; i < num_ext; ++i) {
    const auto &x = exterior[{i, 0}];
    const auto &y = exterior[{i, 1}];
    exterior_center_x += exterior_center_x / num_ext;
    exterior_center_y += exterior_center_y / num_ext;
  }
  for (std::size_t i = 0; i < num_ext; ++i) {
    const auto &x = exterior[{i, 0}] - exterior_center_x;
    const auto &y = exterior[{i, 1}] - exterior_center_y;
    exterior[{i, 0}] = object_x + x * cos_yaw - y * sin_yaw + exterior_center_x;
    exterior[{i, 1}] = object_y + y * cos_yaw + x * sin_yaw + exterior_center_y;
  }

  // find closest point
  Scalar min_dist =
      sqr_distance(tip_x, tip_y, exterior[{0, 0}], exterior[{0, 1}]);
  std::size_t min_i = 0;
  for (std::size_t i = 0; i < num_ext; ++i) {
    Scalar dist =
        sqr_distance(tip_x, tip_y, exterior[{i, 0}], exterior[{i, 1}]);
    if (dist < min_dist) {
      min_dist = dist;
      min_i = i;
    }
  }

  // determine closest edge to the "left" or to the "right" of the closest
  // point (cx, cy)
  std::size_t left_i = (min_i + num_ext - 1) % num_ext;
  std::size_t right_i = (min_i + num_ext + 1) % num_ext;
  const Scalar cx = exterior[{min_i, 0}], cy = exterior[{min_i, 1}];
  const Scalar lx = exterior[{left_i, 0}], ly = exterior[{left_i, 1}];
  const Scalar rx = exterior[{right_i, 0}], ry = exterior[{right_i, 1}];
  Scalar left_line_intersect_x, left_line_intersect_y, left_circle_intersect_x,
      left_circle_intersect_y;
  Scalar right_line_intersect_x, right_line_intersect_y,
      right_circle_intersect_x, right_circle_intersect_y;
  Scalar left_dist = line_distance<Scalar, Utils>(
      tip_x, tip_y, cx, cy, lx, ly, tip_radius, left_line_intersect_x,
      left_line_intersect_y, left_circle_intersect_x, left_circle_intersect_y);
  Scalar right_dist = line_distance<Scalar, Utils>(
      tip_x, tip_y, rx, ry, cx, cy, tip_radius, right_line_intersect_x,
      right_line_intersect_y, right_circle_intersect_x,
      right_circle_intersect_y);

  bool inside = false;
  if (right_dist < left_dist) {
    printf("RIGHT  ");
    // check if point is inside boundary, i.e. "below" the edge
    if ((rx - cx) * (tip_y - cy) - (ry - cy) * (tip_x - cx) < Utils::zero()) {
      inside = true;
    }
    cp.m_distance = right_dist;

    // compute line normal
    cp.m_world_normal_on_b.m_x = cy - ry;
    cp.m_world_normal_on_b.m_y = -(cx - rx);
    cp.m_world_normal_on_b.m_z = Utils::zero();

    cp.m_world_point_on_a.m_x = right_circle_intersect_x;
    cp.m_world_point_on_a.m_y = right_circle_intersect_y;
    cp.m_world_point_on_a.m_z = Utils::zero();

    cp.m_world_point_on_b.m_x = right_line_intersect_x;
    cp.m_world_point_on_b.m_y = right_line_intersect_y;
    cp.m_world_point_on_b.m_z = Utils::zero();
  } else {
    printf("LEFT   ");
    // check if point is inside boundary, i.e. "below" the edge
    if ((cx - lx) * (tip_y - ly) - (cy - ly) * (tip_x - lx) < Utils::zero()) {
      inside = true;
    }
    cp.m_distance = left_dist;

    // compute line normal
    cp.m_world_normal_on_b.m_x = cy - ry;
    cp.m_world_normal_on_b.m_y = -(cx - rx);
    cp.m_world_normal_on_b.m_z = Utils::zero();

    cp.m_world_point_on_a.m_x = left_circle_intersect_x;
    cp.m_world_point_on_a.m_y = left_circle_intersect_y;
    cp.m_world_point_on_a.m_z = Utils::zero();

    cp.m_world_point_on_b.m_x = left_line_intersect_x;
    cp.m_world_point_on_b.m_y = left_line_intersect_y;
    cp.m_world_point_on_b.m_z = Utils::zero();
  }
  if (inside) {
    cp.m_distance = -cp.m_distance;
  }
  cp.m_distance -= tip_radius;
  printf("contact normal: [%.3f %.3f] \t", cp.m_world_normal_on_b.m_x,
         cp.m_world_normal_on_b.m_y);
  printf("inside? %i \t", inside);
  printf("distance: %.3f\n", Utils::getDouble(cp.m_distance));

  return cp;
}

int make_sphere(VisualizerAPI *sim, float r = 1, float g = 0.6f, float b = 0,
                float a = 0.8f) {
  int sphere_id = sim->loadURDF("sphere_small.urdf");
  b3RobotSimulatorChangeVisualShapeArgs vargs;
  vargs.m_objectUniqueId = sphere_id;
  vargs.m_hasRgbaColor = true;
  vargs.m_rgbaColor = btVector4(r, g, b, a);
  sim->changeVisualShape(vargs);
  return sphere_id;
}

void update_position(VisualizerAPI *sim, int object_id, double x, double y,
                     double z) {
  btVector3 pos(x, y, z);
  btQuaternion orn;
  sim->resetBasePositionAndOrientation(object_id, pos, orn);
}

template <typename Scalar, typename Utils>
struct PushExperiment {

};

int main(int argc, char *argv[]) {
  typedef double Scalar;
  typedef DoubleUtils Utils;

  std::string connection_mode = "gui";

  std::string shape = "butter";

  std::string object_filename;
  TinyFileUtils::find_file("mit-push/obj/" + shape + ".urdf", object_filename);
  std::string tip_filename;
  TinyFileUtils::find_file("mit-push/obj/tip.urdf", tip_filename);
  std::string ground_filename;
  TinyFileUtils::find_file("mit-push/obj/plywood.urdf", ground_filename);

  std::string exterior_filename;
  TinyFileUtils::find_file("mit-push/obj/" + shape + "_ext.npy",
                           exterior_filename);
  TinyNumpyReader<Scalar, 2> npy_reader;
  bool npy_success = npy_reader.Open(exterior_filename);
  assert(npy_success);
  auto exterior = npy_reader.Read();

  std::string push_filename;
  TinyFileUtils::find_file(
      "mit-push/abs/" + shape + "/" + shape + "_h5/"
      "motion_surface=abs_shape=" + shape + "_a=0_v=10_i=0.000_s=0.000_t=0.000.h5",
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

  TinyUrdfCache<Scalar, Utils> urdf_cache;

  // create duplicate world object to not get any automatic collision response
  // between object and tip (we create our own contact point for this
  // interaction)
  TinyWorld<Scalar, Utils> world, world2;
  world.set_gravity(TinyVector3<Scalar, Utils>(0., 0., -9.81));

  TinyMultiBody<Scalar, Utils> *ground =
      urdf_cache.construct(ground_filename, world, sim2, sim);
  bool ignore_cache = true;
  TinyMultiBody<Scalar, Utils> *true_object =
      urdf_cache.construct(object_filename, world2, sim2, sim, ignore_cache);
  TinyMultiBody<Scalar, Utils> *object =
      urdf_cache.construct(object_filename, world, sim2, sim, ignore_cache);
  TinyMultiBody<Scalar, Utils> *tip =
      urdf_cache.construct(tip_filename, world2, sim2, sim);

  {
    // set up contact (friction) model for surface <-> object contact
    delete world.m_mb_constraint_solver;
    auto *spring_contact =
        new TinyMultiBodyConstraintSolverSpring<Scalar, Utils>;
    spring_contact->spring_k = 1;
    spring_contact->damper_d = 0.1;
    spring_contact->mu_static = 0.1;
    // spring_contact->friction_model = FRICTION_NONE;
    world.m_mb_constraint_solver = spring_contact;
    world.default_friction = 0.5;
  }

  // Choose solver for contact between tip and object
  TinyMultiBodyConstraintSolver<Scalar, Utils> tip_contact_model;
  // TinyMultiBodyConstraintSolverSpring<Scalar, Utils> tip_contact_model;
  // friction between tip and object from the paper
  world2.default_friction = 0.25;

  fflush(stdout);

  // double dt = 1. / 1000.;
  double time = 0;

  for (auto &link : true_object->m_links) {
    for (auto visual_id : link.m_visual_uids1) {
      b3RobotSimulatorChangeVisualShapeArgs vargs;
      vargs.m_objectUniqueId = visual_id;
      vargs.m_hasRgbaColor = true;
      vargs.m_rgbaColor = btVector4(0.1, 0.6, 0, 0.7);
      sim->changeVisualShape(vargs);
    }
  }

  std::size_t skip_steps = 1;
  Scalar dt = skip_steps * data.dt;

  printf("dt: %.5f\n", Utils::getDouble(dt));

  auto yaw_correction = [](const Scalar& yaw) {
    return yaw + M_PI_2;
  };

  while (sim->isConnected()) {
    printf("Playback...\n");

    object->initialize();
    object->m_q[0] = data.object_x[0];
    object->m_q[1] = data.object_y[0];
    object->m_q[2] = 0.02;
    object->m_q[3] = yaw_correction(data.object_yaw[0]);
    object->forward_kinematics();
    PyBulletUrdfImport<Scalar, Utils>::sync_graphics_transforms(object, *sim);

    for (std::size_t i = 0; i < data.time.size(); i += skip_steps) {
      tip->m_q[0] = data.tip_x[i];
      tip->m_q[1] = data.tip_y[i];
      if (i > 0) {
        tip->m_qd[0] = (data.tip_x[i] - data.tip_x[i - 1]) / data.dt;
        tip->m_qd[1] = (data.tip_y[i] - data.tip_y[i - 1]) / data.dt;
      }
      tip->forward_kinematics();
      PyBulletUrdfImport<Scalar, Utils>::sync_graphics_transforms(tip, *sim);

      object->forward_dynamics(world.get_gravity());
      object->clear_forces();
      auto tip_contact = compute_contact(tip, object, exterior);
      tip_contact_model.resolveCollision({tip_contact}, dt);
      world.step(dt);
      object->integrate(dt);
      object->forward_kinematics();
      PyBulletUrdfImport<Scalar, Utils>::sync_graphics_transforms(object, *sim);

      true_object->m_q[0] = data.object_x[i];
      true_object->m_q[1] = data.object_y[i];
      true_object->m_q[3] = yaw_correction(data.object_yaw[i]);
      true_object->forward_kinematics();
      PyBulletUrdfImport<Scalar, Utils>::sync_graphics_transforms(true_object,
                                                                  *sim);

      std::this_thread::sleep_for(std::chrono::duration<double>(dt));
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(5.));
  }

  delete sim;
  delete sim2;

  return EXIT_SUCCESS;
}
