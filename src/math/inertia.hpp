#pragma once

// #include "enoki_algebra.hpp"
#include "spatial_vector.hpp"

template <typename Algebra>
struct RigidBodyInertia {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;
  typedef ::MotionVector<Algebra> MotionVector;
  typedef ::ForceVector<Algebra> ForceVector;

  /**
   * Mass \f$m\f$.
   */
  Scalar mass{0};

  /**
   * Center of mass, also denoted as \f$h\f$.
   */
  Vector3 com{0};

  Matrix3 inertia{Algebra::diagonal3(1)};

  RigidBodyInertia() = default;

  RigidBodyInertia(const RigidBodyInertia<Algebra> &rbi) = default;

  RigidBodyInertia(const Scalar &mass, const Scalar &Ixx, const Scalar &Iyx,
                   const Scalar &Iyy, const Scalar &Izx, const Scalar &Izy,
                   const Scalar &Izz)
      : mass(mass) {
    inertia = {{Ixx, Iyx, Izx}, {Iyx, Iyy, Izy}, {Izx, Izy, Izz}};
  }

  RigidBodyInertia(const Scalar &mass) : mass(mass) {}

  RigidBodyInertia(const Scalar &mass, const Vector3 &com,
                   const Matrix3 &inertia)
      : mass(mass), com(com), inertia(inertia) {}

  RigidBodyInertia(const Scalar &mass, const Vector3 &com, const Scalar &Ixx,
                   const Scalar &Iyx, const Scalar &Iyy, const Scalar &Izx,
                   const Scalar &Izy, const Scalar &Izz)
      : mass(mass), com(com) {
    inertia = {{Ixx, Iyx, Izx}, {Iyx, Iyy, Izy}, {Izx, Izy, Izz}};
  }

  RigidBodyInertia(const Matrix6 &m)
      : mass(m(3, 3)),
        com{-m(1, 5), m(0, 5), -m(0, 4)},
        inertia(m(0, 0), m(1, 0), m(2, 0), m(0, 1), m(1, 1), m(2, 1), m(0, 2),
                m(1, 2), m(2, 2)) {}

  void set_zero() {
    mass = Algebra::zero();
    Algebra::set_zero(com);
    Algebra::set_zero(inertia);
  }

  /**
   * Represents spatial inertia matrix where the inertia coincide with the given
   * center of mass.
   * @param mass The mass.
   * @param com The center of mass \f$C\f$.
   * @param inertiaC Inertia matrix \f$I_C\f$ at the center of mass \f$C\f$.
   * @return The spatial inertia data structure.
   */
  static RigidBodyInertia from_mass_com_i(const Scalar &mass,
                                          const Vector3 &com,
                                          const Matrix3 &inertiaC) {
    const Matrix3 crossCom = Algebra::cross_matrix(com);
    const Matrix3 I = inertiaC + crossCom * Algebra::transpose(crossCom) * mass;
    return RigidBodyInertia(mass, com * mass, I(0, 0), I(1, 0), I(1, 1),
                            I(2, 0), I(2, 1), I(2, 2));
  }

  /**
   * Represents spatial inertia matrix where the inertia coincide with the given
   * center of mass.
   * @param mass The mass.
   * @param com The center of mass \f$C\f$.
   * @param gyrationRadii Radii of gyration at the center of mass (diagonal of
   * inertia matrix).
   * @return The spatial inertia data structure.
   */
  static RigidBodyInertia from_mass_com_xyz(const Scalar &mass,
                                            const Vector3 &com,
                                            const Vector3 &gyrationRadii) {
    return RigidBodyInertia(mass, com * mass, gyrationRadii(0), 0,
                            gyrationRadii(1), 0, 0, gyrationRadii(2));
  }

  Matrix6 matrix() const {
    Matrix6 m;
    Algebra::assign_block(m, inertia, 0, 0);
    const auto crossCom = Algebra::cross_matrix(com);
    Algebra::assign_block(m, crossCom, 0, 3);
    Algebra::assign_block(m, -crossCom, 3, 0);
    Algebra::assign_block(m, Algebra::diagonal3(mass), 3, 3);
    return m;
  }

  Matrix6 inverse() const {
    // Inverse of a symmetric block matrix
    // according to (4.1) in
    // http://msvlab.hre.ntou.edu.tw/grades/now/inte/Inverse%20&%20Border/border-LuTT.pdf
    Matrix3 Ainv = Algebra::inverse(inertia);
    Matrix3 B = Algebra::cross_matrix(com);
    Matrix3 C = -B;
    Matrix3 D = Algebra::diagonal3(mass);
    Matrix3 Dinv = Algebra::diagonal3(1.0 / mass);
    Matrix3 DCAB = Algebra::inverse(D - C * Ainv * B);
    Matrix3 AinvBDCAB = Ainv * B * DCAB;

    Matrix6 m;
    Algebra::assign_block(m, Ainv + AinvBDCAB * C * Ainv, 0, 0);
    Algebra::assign_block(m, -AinvBDCAB, 0, 3);
    Algebra::assign_block(m, -DCAB * C * Ainv, 3, 0);
    Algebra::assign_block(m, DCAB, 3, 3);
    return m;
  }

  /**
   * V = mv(w, v)
   * I*v = fv(Iw + hxv, mv - hxw)
   */
  ForceVector operator*(const MotionVector &v) const {
    ForceVector result;
    result.top = inertia * v.top + Algebra::cross(com, v.bottom);
    result.bottom = mass * v.bottom - Algebra::cross(com, v.top);
    return result;
  }

  RigidBodyInertia operator+(const RigidBodyInertia &rbi) const {
    return Inertia(mass + rbi.mass, com + rbi.com, inertia + rbi.inertia);
  }

  RigidBodyInertia &operator+=(const RigidBodyInertia &rbi) {
    mass += rbi.mass;
    com += rbi.com;
    inertia += rbi.inertia;
    return *this;
  }

  RigidBodyInertia &operator=(const Matrix6 &m) {
    mass = m(3, 3);
    com = Vector3(-m(1, 5), m(0, 5), -m(0, 4));
    Algebra::assign_block(inertia, m, 0, 0, 3, 3);
    return *this;
  }

  void print(const char *name) const {
    printf("%s\n", name);
    printf("  mass:    %.8f\n", Algebra::to_double(mass));
    printf("  com:     %.8f\t%.8f\t%.8f\n", Algebra::to_double(com[0]),
           Algebra::to_double(com[1]), Algebra::to_double(com[2]));
    printf("  inertia: %.8f\t%.8f\t%.8f\n", Algebra::to_double(inertia(0, 0)),
           Algebra::to_double(inertia(0, 1)),
           Algebra::to_double(inertia(0, 2)));
    printf("           %.8f\t%.8f\t%.8f\n", Algebra::to_double(inertia(1, 0)),
           Algebra::to_double(inertia(1, 1)),
           Algebra::to_double(inertia(1, 2)));
    printf("           %.8f\t%.8f\t%.8f\n", Algebra::to_double(inertia(2, 0)),
           Algebra::to_double(inertia(2, 1)),
           Algebra::to_double(inertia(2, 2)));
  }

  // ENOKI_STRUCT(RigidBodyInertia, mass, com, inertia)
};
// ENOKI_STRUCT_SUPPORT(RigidBodyInertia, mass, com, inertia)

/**
 * The articulated body inertia matrix has the form
 *   [  I   H ]
 *   [ H^T  M ]
 * where M and I are symmetric 3x3 matrices.
 */
template <typename Algebra>
struct ArticulatedBodyInertia {
  using Index = typename Algebra::Index;
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;
  typedef ::MotionVector<Algebra> MotionVector;
  typedef ::ForceVector<Algebra> ForceVector;
  typedef ::RigidBodyInertia<Algebra> RigidBodyInertia;

  Matrix3 I{Algebra::diagonal3(1.)};
  Matrix3 H{Algebra::zero33()};
  Matrix3 M{Algebra::diagonal3(1.)};

  ArticulatedBodyInertia() = default;
  ArticulatedBodyInertia(const Matrix3 &I, const Matrix3 &H, const Matrix3 &M)
      : I(I), H(H), M(M) {}

  ArticulatedBodyInertia(const RigidBodyInertia &rbi)
      : I(rbi.inertia),
        H(Algebra::cross_matrix(rbi.com)),
        M(Algebra::diagonal3(rbi.mass)) {}

  ArticulatedBodyInertia &operator=(const RigidBodyInertia &rbi) {
    I = rbi.inertia;
    H = Algebra::cross_matrix(rbi.com);
    M = Algebra::diagonal3(rbi.mass);
    return *this;
  }

  Matrix6 matrix() const {
    Matrix6 m;
    Algebra::assign_block(m, I, 0, 0);
    Algebra::assign_block(m, H, 0, 3);
    Algebra::assign_block(m, Algebra::transpose(H), 3, 0);
    Algebra::assign_block(m, M, 3, 3);
    return m;
  }

  /**
   * V = mv(w, v)
   * Ia*v = mv(Iw + Hv, Mv + H^T w)
   */
  ForceVector operator*(const MotionVector &v) const {
    ForceVector result;
    result.top = I * v.top + H * v.bottom;
    result.bottom = M * v.bottom + Algebra::transpose(H) * v.top;
    return result;
  }

  ArticulatedBodyInertia operator+(const ArticulatedBodyInertia &abi) const {
    return ArticulatedBodyInertia(I + abi.I, H + abi.H, M + abi.M);
  }
  ArticulatedBodyInertia operator-(const ArticulatedBodyInertia &abi) const {
    return ArticulatedBodyInertia(I - abi.I, H - abi.H, M - abi.M);
  }

  ArticulatedBodyInertia &operator+=(const ArticulatedBodyInertia &abi) {
    I += abi.I;
    H += abi.H;
    M += abi.M;
    return *this;
  }

  ArticulatedBodyInertia &operator+=(const Matrix6 &m) {
    Matrix3 tmp;
    Algebra::assign_block(tmp, m, 0, 0, 3, 3, 0, 0);
    I += tmp;
    Algebra::assign_block(tmp, m, 0, 0, 3, 3, 0, 3);
    H += tmp;
    Algebra::assign_block(tmp, m, 0, 0, 3, 3, 3, 3);
    M += tmp;
    return *this;
  }

  ArticulatedBodyInertia &operator-=(const ArticulatedBodyInertia &abi) {
    I -= abi.I;
    H -= abi.H;
    M -= abi.M;
    return *this;
  }

  ArticulatedBodyInertia &operator-=(const Matrix6 &m) {
    Matrix3 tmp;
    Algebra::assign_block(tmp, m, 0, 0, 3, 3, 0, 0);
    I -= tmp;
    Algebra::assign_block(tmp, m, 0, 0, 3, 3, 0, 3);
    H -= tmp;
    Algebra::assign_block(tmp, m, 0, 0, 3, 3, 3, 3);
    M -= tmp;
    return *this;
  }

  ArticulatedBodyInertia operator+(const Matrix6 &m) const {
    ArticulatedBodyInertia abi(*this);
    abi += m;
    return abi;
  }

  ArticulatedBodyInertia operator-(const Matrix6 &m) const {
    ArticulatedBodyInertia abi(*this);
    abi -= m;
    return abi;
  }

  ArticulatedBodyInertia &operator=(const Matrix6 &m) {
    Algebra::assign_block(I, m, 0, 0, 3, 3, 0, 0);
    Algebra::assign_block(H, m, 0, 0, 3, 3, 0, 3);
    Algebra::assign_block(M, m, 0, 0, 3, 3, 3, 3);
    return *this;
  }

  // Matrix6 inverse() const {
  //   // Inverse of a symmetric block matrix
  //   // according to (4.1) in
  //   //
  //   //
  //   http://msvlab.hre.ntou.edu.tw/grades/now/inte/Inverse%20&%20Border/border-LuTT.pdf
  //   Matrix3 Ainv = Algebra::inverse(I);
  //   Matrix3 B = H;
  //   Matrix3 C = -B;
  //   Matrix3 DCAB = Algebra::inverse(M - C * Ainv * B);
  //   Matrix3 AinvBDCAB = Ainv * B * DCAB;

  //   Matrix6 m;
  //   Algebra::assign_block(m, Ainv + AinvBDCAB * C * Ainv, 0, 0);
  //   Algebra::assign_block(m, -AinvBDCAB, 0, 3);
  //   Algebra::assign_block(m, -DCAB * C * Ainv, 3, 0);
  //   Algebra::assign_block(m, DCAB, 3, 3);
  //   return m;
  // }

  ArticulatedBodyInertia inverse() const {
    // Inverse of a symmetric block matrix
    // according to (4.1) in
    //
    // http://msvlab.hre.ntou.edu.tw/grades/now/inte/Inverse%20&%20Border/border-LuTT.pdf
    Matrix3 Ainv = Algebra::inverse(I);
    Matrix3 B = H;
    Matrix3 C = -B;
    Matrix3 DCAB = Algebra::inverse(M - C * Ainv * B);
    Matrix3 AinvBDCAB = Ainv * B * DCAB;

    ArticulatedBodyInertia abi;
    abi.I = Ainv + AinvBDCAB * C * Ainv;
    abi.H = -AinvBDCAB;
    abi.M = DCAB;
    return abi;
  }

  MotionVector inv_mul(const ForceVector &v) const {
    // TODO verify
    ArticulatedBodyInertia abi = inverse();
    MotionVector result;
    result.top = abi.I * v.top + abi.H * v.bottom;
    result.bottom = abi.M * v.bottom + Algebra::transpose(abi.H) * v.top;
    return result;
  }

  /**
   * Multiplies force vectors a and b as a * b^T, resulting in a 6x6 matrix.
   */
  static ArticulatedBodyInertia mul_transpose(const ForceVector &a,
                                              const ForceVector &b) {
    // printf("mul_transpose:\n");
    // Algebra::print("a", a);
    // Algebra::print("b", b);
    ArticulatedBodyInertia abi;
    for (Index i = 0; i < 3; i++) {
      Algebra::assign_column(abi.I, i, a.top * b.top[i]);
      Algebra::assign_column(abi.H, i, a.top * b.bottom[i]);
      Algebra::assign_column(abi.M, i, a.bottom * b.bottom[i]);
    }
    return abi;
  }

  void print(const char *name) const {
    printf("%s\n", name);
    print("I", I, 4);
    print("H", H, 4);
    print("M", M, 4);
  }

 private:
  static void print(const char *name, const Matrix3 &m, int indent) {
    for (int i = 0; i < indent; ++i) {
      printf(" ");
    }
    printf("%s:\n", name);
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < indent; ++i) {
        printf(" ");
      }
      printf("%.8f  %.8f  %.8f\n", m(j, 0), m(j, 1), m(j, 2));
    }
  }

  // ENOKI_STRUCT(ArticulatedBodyInertia, I, H, M)
};
// ENOKI_STRUCT_SUPPORT(ArticulatedBodyInertia, I, H, M)
