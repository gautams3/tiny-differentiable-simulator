#ifndef NEURAL_PUSH_UTILS_H
#define NEURAL_PUSH_UTILS_H

#include <highfive/H5Easy.hpp>
#include <regex>
#include <vector>

#include "tiny_dataset.h"
#include "tiny_mb_constraint_solver_spring.h"
#include "tiny_multi_body.h"

template <typename Scalar>
Scalar yaw_correction(const Scalar &yaw) {
  return yaw + Scalar(M_PI_2);
}

struct PushData {
  std::vector<double> time;
  std::vector<double> tip_x, tip_y, tip_yaw;
  std::vector<double> object_x, object_y, object_yaw;
  std::vector<double> force_x, force_y, force_yaw;

  std::vector<std::vector<double>> states;

  double dt{0};

  struct Entry {
    double tip_x, tip_y, tip_yaw;
    double object_x, object_y, object_yaw;
    double force_x, force_y, force_yaw;
  };

  std::string filename;
  std::string shape_name;
  std::string surface_name;

  std::string lab_name;

  PushData(const std::string &filename) : filename(filename) {
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
      tip_yaw.push_back(yaw_correction(vec[3]));
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
      object_yaw.push_back(yaw_correction(vec[3]));
    }
    vecs.clear();

    data = push_file.getDataSet("ft_wrench");
    data.read(vecs);
    for (std::size_t i = 0; i < vecs.size(); ++i) {
      const auto &vec = vecs[i];
      force_x.push_back(vec[1]);
      force_y.push_back(vec[2]);
      force_yaw.push_back(yaw_correction(vec[3]));
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

    // generate state vectors for the cost computation
    states.resize(min_len);
    for (std::size_t i = 0; i < min_len; ++i) {
      states[i] = {object_x[i], object_y[i], object_yaw[i]};
    }

    static const std::regex object_regex(".*shape=([a-z0-9]+).*");
    std::smatch match;
    std::regex_match(filename, match, object_regex);
    shape_name = match[1].str();

    static const std::regex surface_regex(".*surface=([a-z0-9]+).*");
    std::regex_match(filename, match, surface_regex);
    surface_name = match[1].str();

    lab_name = surface_name + shape_name;

    std::cout << "Read push dataset \"" + filename + "\" with " << tip_x.size()
              << " entries.\n\tTime step: " << dt << "\n\t"
              << "Shape:     " << shape_name << "\n\t"
              << "Surface:   " << surface_name << "\n";
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
 * Rotates points by yaw angle around their centroid, and moves them by dx and
 * dy.
 */
template <typename Scalar, typename Utils>
void transform_points(const TinyDataset<double, 2> &exterior,
                      TinyDataset<Scalar, 2> &tf_exterior, const Scalar &dx,
                      const Scalar &dy, const Scalar &yaw) {
  const Scalar cos_yaw = Utils::cos1(yaw), sin_yaw = Utils::sin1(yaw);
  double exterior_center_x = 0;
  double exterior_center_y = 0;
  const std::size_t num_ext = exterior.Shape()[0];
  double dnum_ext = num_ext;
  for (std::size_t i = 0; i < num_ext; ++i) {
    const auto &x = exterior[{i, 0}];
    const auto &y = exterior[{i, 1}];
    exterior_center_x += exterior_center_x / dnum_ext;
    exterior_center_y += exterior_center_y / dnum_ext;
  }
  for (std::size_t i = 0; i < num_ext; ++i) {
    const Scalar x = Scalar(exterior[{i, 0}] - exterior_center_x);
    const Scalar y = Scalar(exterior[{i, 1}] - exterior_center_y);
    tf_exterior[{i, 0}] = dx + x * cos_yaw - y * sin_yaw + exterior_center_x;
    tf_exterior[{i, 1}] = dy + y * cos_yaw + x * sin_yaw + exterior_center_y;
  }
}

/**
 * Computes a contact point between the tip (circle) and a 2D polygon described
 * by its edges in a contiguous array of points.
 *
 *    Multi body `a` is the tip.
 *    Multi body `b` is the object.
 *
 * The exterior points are assumed to be transformed already by the object pose.
 */
template <typename Scalar, typename Utils>
TinyContactPointMultiBody<Scalar, Utils> compute_contact(
    TinyMultiBody<Scalar, Utils> *tip, TinyMultiBody<Scalar, Utils> *object,
    const TinyDataset<Scalar, 2> &exterior,
    const Scalar &tip_radius = Scalar(0.0045)) {
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
#if DEBUG
    printf("RIGHT  ");
#endif
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
#if DEBUG
    printf("LEFT   ");
#endif
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
#if DEBUG
  printf("contact normal: [%.3f %.3f] \t",
         Utils::getDouble(cp.m_world_normal_on_b.m_x),
         Utils::getDouble(cp.m_world_normal_on_b.m_y));
  printf("inside? %i \t", inside);
  printf("distance: %.3f\n", Utils::getDouble(cp.m_distance));
#endif

  return cp;
}

#endif  // NEURAL_PUSH_UTILS_H
