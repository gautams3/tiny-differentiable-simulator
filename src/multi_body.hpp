#pragma once

#include "link.hpp"

template <typename Algebra>
class MultiBody {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using VectorX = typename Algebra::VectorX;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;
  using Quaternion = typename Algebra::Quaternion;
  typedef ::Transform<Algebra> Transform;
  typedef ::MotionVector<Algebra> MotionVector;
  typedef ::ForceVector<Algebra> ForceVector;
  typedef ::Link<Algebra> Link;
  typedef ::RigidBodyInertia<Algebra> RigidBodyInertia;
  typedef ::ArticulatedBodyInertia<Algebra> ArticulatedBodyInertia;

  /**
   * Number of degrees of freedom, excluding floating-base coordinates.
   */
  int dof_{0};

 public:
  std::vector<Link> links;

  /**
   * Dimensionality of joint positions q (including 7-DoF floating-base
   * coordinates if this system is floating-base).
   */
  TINY_INLINE int dof() const { return is_floating ? dof_ + 7 : dof_; }
  /**
   * Dimensionality of joint velocities qd and accelerations qdd (including
   * 6-DoF base velocity and acceleration, if this system is floating-base).
   */
  TINY_INLINE int dof_qd() const { return is_floating ? dof_ + 6 : dof_; }

  /**
   * Indices in `tau` that are controllable, i.e. actuated.
   * For floating-base system, the index 0 corresponds to the first degree of
   * freedom not part of the 6D floating-base coordinates.
   */
  std::vector<int> control_indices;
  /**
   * Dimensionality of control input, i.e. number of actuated DOFs.
   */
  TINY_INLINE int dof_actuated() const {
    return static_cast<int>(control_indices.size());
  }

  /**
   * Whether this system is floating or fixed to the world frame.
   */
  bool is_floating{false};

  // quantities related to floating base
  mutable MotionVector base_velocity;       // v_0
  mutable MotionVector base_acceleration;   // a_0
  ForceVector base_applied_force;           // f_ext_0 in world frame
  mutable ForceVector base_force;           // f_0 (used by RNEA)
  mutable ForceVector base_bias_force;      // pA_0
  RigidBodyInertia base_rbi;                // I_0
  mutable ArticulatedBodyInertia base_abi;  // IA_0
  mutable Transform base_X_world;

  std::vector<int> visual_uids1;
  std::vector<int> visual_uids2;
  // offset of geometry (relative to the base frame)
  std::vector<Transform> X_visuals;

  // std::vector<const TinyGeometry<Scalar, Algebra> *>
  //     collision_geometries;
  // offset of collision geometries (relative to this link frame)
  std::vector<Transform> X_collisions;

  VectorX q, qd, qdd, tau;

  explicit MultiBody(bool isFloating = false) : is_floating(isFloating) {}

  /**
   * Set 3D base position in world coordinates.
   */
  void set_position(const Vector3 &initial_position) {
    base_X_world.translation = initial_position;
    if (is_floating) {
      q[4] = initial_position[0];
      q[5] = initial_position[1];
      q[6] = initial_position[2];
    }
  }

  /**
   * Ensures that the joint coordinates q, qd, qdd, tau are initialized
   * properly in the MultiBody member variables.
   */
  void initialize() {
    // make sure dof and the q / qd indices in the links are accurate
    int q_index = is_floating ? 7 : 0;
    int qd_index = is_floating ? 6 : 0;
    dof_ = 0;  // excludes floating-base DOF
    for (Link &link : links) {
      assert(link.index >= 0);
      link.q_index = q_index;
      link.qd_index = qd_index;
      if (link.joint_type != JOINT_FIXED) {
        ++q_index;
        ++qd_index;
        ++dof_;
      } else {
        link.q_index = -2;
        link.qd_index = -2;
      }
    }

    q = Algebra::zerox(dof());
    qd = Algebra::zerox(dof_qd());
    qdd = Algebra::zerox(dof_qd());
    tau = Algebra::zerox(dof_actuated());
    if (is_floating) {
      q[3] = Algebra::one();  // make sure orientation is valid
    }

    // (Re-)create actuator to make sure it has the right degrees of freedom.
    // if (actuator) {
    //   delete actuator;
    //   actuator = new TinyActuator(dof_actuated());
    // }
  }

  /**
   * Copy constructor. Skips visualization members, temporary variables.
   * The actuator is not copied, but the original pointer `m_actuator` is
   * carried over.
   */
  template <typename Algebra2>
  MultiBody(const MultiBody<Algebra2> &mb)
      : links(mb.links),
        dof_(mb.dof_),
        // actuator(mb.actuator),
        control_indices(mb.control_indices),
        is_floating(mb.is_floating),
        base_velocity(mb.base_velocity),
        base_acceleration(mb.base_acceleration),
        base_applied_force(mb.base_applied_force),
        base_force(mb.base_force),
        base_bias_force(mb.base_bias_force),
        base_rbi(mb.base_rbi),
        base_X_world(mb.base_X_world),
        // collision_geometries(mb.collision_geometries),
        X_collisions(mb.X_collisions),
        q(mb.q),
        qd(mb.qd),
        qdd(mb.qdd),
        tau(mb.tau) {}

  // virtual ~MultiBody() {
  // if (m_actuator) {
  //   delete actuator;
  // }
  // }

  void print_state() const {
    printf("q: [");
    for (int i = 0; i < dof(); ++i) {
      if (i > 0) printf(" ");
      printf("%.2f", Algebra::to_double(q[i]));
    }
    printf("] \tqd: [");
    for (int i = 0; i < dof_qd(); ++i) {
      if (i > 0) printf(" ");
      printf("%.2f", Algebra::to_double(qd[i]));
    }
    printf("] \tqdd: [");
    for (int i = 0; i < dof_qd(); ++i) {
      if (i > 0) printf(" ");
      printf("%.2f", Algebra::to_double(qdd[i]));
    }
    printf("] \ttau: [");
    for (int i = 0; i < dof_actuated(); ++i) {
      if (i > 0) printf(" ");
      printf("%.2f", Algebra::to_double(tau[i]));
    }
    printf("]\n");
  }

  const Transform &get_world_transform(int link) const {
    if (link == -1) {
      return base_X_world;
    } else {
      return links[link].X_world;
    }
  }

  /**
   * Compute center of mass of link in world coordinates.
   * @param link Index of link in `links`.
   * @return 3D coordinates of center of mass in world coordinates.
   */
  const Vector3 get_world_com(int link) const {
    const Transform &tf = get_world_transform(link);
    if (link == -1) {
      return tf.apply(base_rbi.com);
    } else {
      return tf.apply(links[link].I.com);
    }
  }

  TINY_INLINE Scalar get_q_for_link(const VectorX &q, int link_index) const {
    if (Algebra::size(q) == 0) return Algebra::zero();
    const Link &link = links[link_index];
    return link.joint_type == JOINT_FIXED ? Algebra::zero() : q[link.q_index];
  }
  TINY_INLINE Scalar get_q_for_link(int link_index) const {
    get_q_for_link(q, link_index);
  }

  TINY_INLINE Scalar get_qd_for_link(const VectorX &qd, int link_index) const {
    if (Algebra::size(qd) == 0) return Algebra::zero();
    const Link &link = links[link_index];
    return link.joint_type == JOINT_FIXED ? Algebra::zero() : qd[link.qd_index];
  }
  TINY_INLINE Scalar get_qd_for_link(int link_index) const {
    return get_qd_for_link(qd, link_index);
  }

  TINY_INLINE Scalar get_qdd_for_link(const VectorX &qdd,
                                      int link_index) const {
    return get_qd_for_link(qdd, link_index);
  }
  TINY_INLINE Scalar get_qdd_for_link(int link_index) const {
    return get_qdd_for_link(qdd, link_index);
  }

  TINY_INLINE Scalar get_tau_for_link(const VectorX &tau,
                                      int link_index) const {
    if (Algebra::size(tau) == 0) return Algebra::zero();
    const Link &link = links[link_index];
    int offset = is_floating ? -6 : 0;
    return link.joint_type == JOINT_FIXED ? Algebra::zero()
                                          : tau[link.qd_index + offset];
  }
  TINY_INLINE Scalar get_tau_for_link(int link_index) const {
    return get_tau_for_link(tau, link_index);
  }

  /**
   * Set joint torques and external forces in all links and the base to zero.
   */
  void clear_forces() {
    base_applied_force.set_zero();
    for (Link &link : links) {
      link.f_ext.set_zero();
    }
    for (int i = 0; i < dof_actuated(); ++i) {
      tau[i] = Algebra::zero();
    }
  }

  /**
   * Implements the first phase in ABA, CRBA and RNEA, that computes the
   * joint and body transforms, velocities and bias forces.
   * Initializes articulated inertia with the local body inertia.
   *
   * Joint positions q must have dimension of dof().
   * Joint velocities qd must have dimension of dof_qd().
   * If no joint velocities qd are given, qd is assumed to be zero.
   * If no joint accelerations qdd are given, qdd is assumed to be zero.
   */
  void forward_kinematics(const VectorX &q, const VectorX &qd = VectorX(),
                          const VectorX &qdd = VectorX()) const;

  /**
   * Updates the forward kinematics given the q, qd coordinates stored in this
   * model.
   */
  void forward_kinematics() { forward_kinematics(q, qd); }

  void forward_dynamics(const VectorX &q, const VectorX &qd, const VectorX &tau,
                        const Vector3 &gravity, VectorX &qdd) const;

  void forward_dynamics(const Vector3 &gravity) {
    forward_dynamics(q, qd, tau, gravity, qdd);
  }

  static std::string joint_type_name(JointType t) {
    static std::string names[] = {
        "JOINT_FIXED",       "JOINT_PRISMATIC_X",    "JOINT_PRISMATIC_Y",
        "JOINT_PRISMATIC_Z", "JOINT_PRISMATIC_AXIS", "JOINT_REVOLUTE_X",
        "JOINT_REVOLUTE_Y",  "JOINT_REVOLUTE_Z",     "JOINT_REVOLUTE_AXIS",
    };
    return names[int(t) + 1];
  }

  // attaches a new link, setting parent to the last link
  void attach(Link &link, bool is_controllable = true) {
    int parent_index = -1;
    if (!links.empty()) parent_index = static_cast<int>(links.size()) - 1;
    attach(link, parent_index, is_controllable);
  }

  void attach(Link &link, int parent_index, bool is_controllable = true) {
    int sz = static_cast<int>(links.size());
    assert(parent_index < sz);
    link.index = sz;
    link.parent_index = parent_index;
    if (link.joint_type != JOINT_FIXED) {
      assert(Algebra::norm(link.S) > Algebra::zero());
      link.q_index = dof();
      link.qd_index = dof_qd();
      dof_++;
      if (is_controllable) {
        if (control_indices.empty()) {
          control_indices.push_back(0);
        } else {
          control_indices.push_back(control_indices.back() + 1);
        }
      }
    } else {
      link.q_index = -2;
      link.qd_index = -2;
    }
#ifdef DEBUG
    printf(
        "Attached link %i of type %s (parent: %i, index q: %i, index qd: "
        "%i).\n",
        link.index, joint_type_name(link.joint_type).c_str(), link.parent_index,
        link.q_index, link.qd_index);
//    link.S.print("joint.S");
#endif
    links.push_back(link);
  }

  void integrate(VectorX &q, VectorX &qd, VectorX &qdd, const Scalar &dt) {
    assert(static_cast<int>(q.size()) == dof());
    assert(static_cast<int>(qd.size()) == dof_qd());
    assert(static_cast<int>(qdd.size()) == dof_qd());

    int q_offset, qd_offset;
    if (is_floating) {
      base_acceleration.top = Vector3(qdd[0], qdd[1], qdd[2]);
      base_acceleration.bottom = Vector3(qdd[3], qdd[4], qdd[5]);

      base_velocity.top = Vector3(qd[0], qd[1], qd[2]);
      base_velocity.bottom = Vector3(qd[3], qd[4], qd[5]);

      base_velocity += base_acceleration * dt;

      Vector3 linear_velocity = base_velocity.bottom;
      base_X_world.translation += linear_velocity * dt;

      // update base orientation using Quaternion derivative
      Vector3 angular_velocity = base_velocity.top;

      Quaternion base_rot = Algebra::matrix_to_quat(base_X_world.rotation);
      // update 4-dimensional q from 3-dimensional qd for the base rotation
      // angular_velocity = Vector3(qd[0], qd[1], qd[2]);
      base_rot += Algebra::quat_velocity(base_rot, angular_velocity, dt);
      Algebra::normalize(base_rot);
      base_X_world.rotation = Algebra::quat_to_matrix(base_rot);

      q[0] = Algebra::quat_x(base_rot);
      q[1] = Algebra::quat_y(base_rot);
      q[2] = Algebra::quat_z(base_rot);
      q[3] = Algebra::quat_w(base_rot);
      q_offset = 4;
      qd_offset = 3;
    } else {
      q_offset = 0;
      qd_offset = 0;
    }

    for (int i = 0; i < dof_qd() - qd_offset; i++) {
      int qindex = i + q_offset;
      int qdindex = i + qd_offset;
      qd[qdindex] += qdd[qdindex] * dt;
      q[qindex] += qd[qdindex] * dt;
    }
  }

  void integrate(const Scalar &dt) { integrate(q, qd, qdd, dt); }
};

#include "dynamics/forward_dynamics.hpp"
#include "dynamics/kinematics.hpp"