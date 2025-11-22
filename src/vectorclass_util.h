/******************************************************************************
 *
 * Copyright 2025 Bernhard Braun
 *
 * Simplified SIMD helper layer that mimics the subset of the original
 * vectorclass API that is used inside this codebase. The implementation is
 * built on top of SIMDe so it runs on Apple Silicon without x86 SIMD support.
 *
 ******************************************************************************/
#ifndef VECTORCLASS_UTIL_H
#define VECTORCLASS_UTIL_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>

#include "simd_types.h"

// Forward declarations
struct Vec4f;
struct Vec4i;
struct Vec8f;
struct Vec8i;
struct Vec4d;

///////////////////////////////////////////////////////////////////////////////
// 4-wide vectors
///////////////////////////////////////////////////////////////////////////////
struct Vec4i {
  simde__m128i v;
  Vec4i() : v(simde_mm_setzero_si128()) {}
  Vec4i(int32_t x, int32_t y, int32_t z, int32_t w) : v(simde_mm_set_epi32(w, z, y, x)) {}
  Vec4i(int32_t s) : v(simde_mm_set1_epi32(s)) {}
  Vec4i(simde__m128i m) : v(m) {}
  Vec4i& load(const void* p) {
    v = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(p));
    return *this;
  }
  Vec4i& load_a(const void* p) {
    v = simde_mm_load_si128(reinterpret_cast<const simde__m128i*>(p));
    return *this;
  }
  Vec4i& load_partial(int n, const int32_t* p) {
    int32_t tmp[4] = {0, 0, 0, 0};
    for (int i = 0; i < n && i < 4; ++i) tmp[i] = p[i];
    v = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(tmp));
    return *this;
  }
  void store(void* p) const { simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(p), v); }
  void store_a(void* p) const { simde_mm_store_si128(reinterpret_cast<simde__m128i*>(p), v); }
  Vec4i& operator=(int32_t s) { v = simde_mm_set1_epi32(s); return *this; }
  Vec4i& operator=(simde__m128i m) { v = m; return *this; }
  int32_t operator[](int idx) const { int32_t tmp[4]; store(tmp); return tmp[idx]; }
  operator simde__m128i() const { return v; }
};
using Vec4ui = Vec4i;

struct Vec4f {
  simde__m128 v;
  Vec4f() : v(simde_mm_setzero_ps()) {}
  Vec4f(float x, float y, float z, float w) : v(simde_mm_set_ps(w, z, y, x)) {}
  Vec4f(float s) : v(simde_mm_set1_ps(s)) {}
  Vec4f(simde__m128 m) : v(m) {}
  explicit Vec4f(const Vec4i& i) : v(simde_mm_castsi128_ps(i.v)) {}
  Vec4f& load(const float* p) {
    v = simde_mm_loadu_ps(p);
    return *this;
  }
  Vec4f& load_a(const float* p) {
    v = simde_mm_load_ps(p);
    return *this;
  }
  Vec4f& load_partial(int n, const float* p) {
    float tmp[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i = 0; i < n && i < 4; ++i) tmp[i] = p[i];
    v = simde_mm_loadu_ps(tmp);
    return *this;
  }
  void store(float* p) const { simde_mm_storeu_ps(p, v); }
  void store_a(float* p) const { simde_mm_store_ps(p, v); }
  void store_partial(int n, float* p) const {
    float tmp[4];
    simde_mm_storeu_ps(tmp, v);
    for (int i = 0; i < n && i < 4; ++i) p[i] = tmp[i];
  }
  Vec4f& operator=(float s) { v = simde_mm_set1_ps(s); return *this; }
  float operator[](int idx) const { float tmp[4]; store(tmp); return tmp[idx]; }
  void cutoff(int n) {
    float tmp[4];
    store(tmp);
    for (int i = n; i < 4; ++i) tmp[i] = 0.0f;
    load(tmp);
  }
  operator simde__m128() const { return v; }
};

///////////////////////////////////////////////////////////////////////////////
// 8-wide vectors
///////////////////////////////////////////////////////////////////////////////
struct Vec8i {
  simde__m256i v;
  Vec8i() : v(simde_mm256_setzero_si256()) {}
  Vec8i(int32_t a0, int32_t a1, int32_t a2, int32_t a3, int32_t a4, int32_t a5, int32_t a6, int32_t a7)
      : v(simde_mm256_set_epi32(a7, a6, a5, a4, a3, a2, a1, a0)) {}
  Vec8i(int32_t s) : v(simde_mm256_set1_epi32(s)) {}
  Vec8i(const Vec4i& lo, const Vec4i& hi)
      : v(simde_mm256_set_m128i(hi.v, lo.v)) {}
  Vec8i(simde__m256i m) : v(m) {}
  Vec8i& load(const void* p) {
    v = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(p));
    return *this;
  }
  Vec8i& load_a(const void* p) {
    v = simde_mm256_load_si256(reinterpret_cast<const simde__m256i*>(p));
    return *this;
  }
  Vec8i& load_partial(int n, const int32_t* p) {
    int32_t tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < n && i < 8; ++i) tmp[i] = p[i];
    v = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(tmp));
    return *this;
  }
  void store(void* p) const { simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(p), v); }
  void store_a(void* p) const { simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(p), v); }
  Vec4i get_low() const { return Vec4i(simde_mm256_castsi256_si128(v)); }
  Vec4i get_high() const { return Vec4i(simde_mm256_extracti128_si256(v, 1)); }
  Vec8i& operator=(int32_t s) { v = simde_mm256_set1_epi32(s); return *this; }
  Vec8i& operator=(simde__m256i m) { v = m; return *this; }
  int32_t operator[](int idx) const { int32_t tmp[8]; store(tmp); return tmp[idx]; }
  operator simde__m256i() const { return v; }
};
using Vec8ui = Vec8i;

struct Vec8f {
  simde__m256 v;
  Vec8f() : v(simde_mm256_setzero_ps()) {}
  Vec8f(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7)
      : v(simde_mm256_set_ps(a7, a6, a5, a4, a3, a2, a1, a0)) {}
  Vec8f(float s) : v(simde_mm256_set1_ps(s)) {}
  Vec8f(simde__m256 m) : v(m) {}
  Vec8f(const Vec4f& lo, const Vec4f& hi) : v(simde_mm256_set_m128(hi.v, lo.v)) {}
  Vec8f& load(const float* p) {
    v = simde_mm256_loadu_ps(p);
    return *this;
  }
  Vec8f& load_a(const float* p) {
    v = simde_mm256_load_ps(p);
    return *this;
  }
  Vec8f& load_partial(int n, const float* p) {
    float tmp[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int i = 0; i < n && i < 8; ++i) tmp[i] = p[i];
    v = simde_mm256_loadu_ps(tmp);
    return *this;
  }
  void store(float* p) const { simde_mm256_storeu_ps(p, v); }
  void store_a(float* p) const { simde_mm256_store_ps(p, v); }
  void store_partial(int n, float* p) const {
    float tmp[8];
    simde_mm256_storeu_ps(tmp, v);
    for (int i = 0; i < n && i < 8; ++i) p[i] = tmp[i];
  }
  Vec4f get_low() const { return Vec4f(simde_mm256_castps256_ps128(v)); }
  Vec4f get_high() const { return Vec4f(simde_mm256_extractf128_ps(v, 1)); }
  Vec8f& operator=(float s) { v = simde_mm256_set1_ps(s); return *this; }
  float operator[](int idx) const { float tmp[8]; store(tmp); return tmp[idx]; }
  void cutoff(int n) {
    float tmp[8];
    store(tmp);
    for (int i = n; i < 8; ++i) tmp[i] = 0.0f;
    load(tmp);
  }
  operator simde__m256() const { return v; }
};

///////////////////////////////////////////////////////////////////////////////
// 64-bit and 16-bit helper vectors
///////////////////////////////////////////////////////////////////////////////
struct Vec8us {
  simde__m128i v;
  Vec8us() : v(simde_mm_setzero_si128()) {}
  Vec8us(uint16_t a0, uint16_t a1, uint16_t a2, uint16_t a3,
         uint16_t a4, uint16_t a5, uint16_t a6, uint16_t a7) {
    uint16_t tmp[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    v = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(tmp));
  }
  explicit Vec8us(simde__m128i m) : v(m) {}
  Vec8us& load(const uint16_t* p) {
    v = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(p));
    return *this;
  }
  Vec8us& load_a(const uint16_t* p) {
    v = simde_mm_load_si128(reinterpret_cast<const simde__m128i*>(p));
    return *this;
  }
  void store(void* p) const { simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(p), v); }
  void store_a(void* p) const { simde_mm_store_si128(reinterpret_cast<simde__m128i*>(p), v); }
  operator simde__m128i() const { return v; }
};

struct Vec4uq {
  simde__m256i v;
  Vec4uq() : v(simde_mm256_setzero_si256()) {}
  Vec4uq(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3)
      : v(simde_mm256_set_epi64x(static_cast<int64_t>(a3),
                                 static_cast<int64_t>(a2),
                                 static_cast<int64_t>(a1),
                                 static_cast<int64_t>(a0))) {}
  explicit Vec4uq(uint64_t s) : v(simde_mm256_set1_epi64x(static_cast<int64_t>(s))) {}
  explicit Vec4uq(simde__m256i m) : v(m) {}
  Vec4uq& load(const uint64_t* p) {
    v = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(p));
    return *this;
  }
  Vec4uq& load_a(const uint64_t* p) {
    v = simde_mm256_load_si256(reinterpret_cast<const simde__m256i*>(p));
    return *this;
  }
  void store(uint64_t* p) const { simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(p), v); }
  void store_a(uint64_t* p) const { simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(p), v); }
  operator simde__m256i() const { return v; }
};

///////////////////////////////////////////////////////////////////////////////
// 4-wide doubles
///////////////////////////////////////////////////////////////////////////////
struct Vec4d {
  simde__m256d v;
  Vec4d() : v(simde_mm256_setzero_pd()) {}
  Vec4d(double x, double y, double z, double w) : v(simde_mm256_set_pd(w, z, y, x)) {}
  Vec4d(double s) : v(simde_mm256_set1_pd(s)) {}
  Vec4d(simde__m256d m) : v(m) {}
  explicit Vec4d(const Vec4f& f) : v(simde_mm256_cvtps_pd(f.v)) {}
  Vec4d& load(const double* p) {
    v = simde_mm256_loadu_pd(p);
    return *this;
  }
  Vec4d& load_a(const double* p) {
    v = simde_mm256_load_pd(p);
    return *this;
  }
  Vec4d& load_partial(int n, const double* p) {
    double tmp[4] = {0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < n && i < 4; ++i) tmp[i] = p[i];
    v = simde_mm256_loadu_pd(tmp);
    return *this;
  }
  void store(double* p) const { simde_mm256_storeu_pd(p, v); }
  void store_a(double* p) const { simde_mm256_store_pd(p, v); }
  void store_partial(int n, double* p) const {
    double tmp[4];
    simde_mm256_storeu_pd(tmp, v);
    for (int i = 0; i < n && i < 4; ++i) p[i] = tmp[i];
  }
  Vec4d& operator=(double s) { v = simde_mm256_set1_pd(s); return *this; }
  double operator[](int idx) const { double tmp[4]; store(tmp); return tmp[idx]; }
  operator simde__m256d() const { return v; }
};

///////////////////////////////////////////////////////////////////////////////
// Arithmetic helpers (float)
///////////////////////////////////////////////////////////////////////////////
inline Vec4f operator+(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_add_ps(a.v, b.v)); }
inline Vec4f operator-(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_sub_ps(a.v, b.v)); }
inline Vec4f operator*(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_mul_ps(a.v, b.v)); }
inline Vec4f operator/(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_div_ps(a.v, b.v)); }
inline Vec4f operator+(const Vec4f& a, float s) { return a + Vec4f(s); }
inline Vec4f operator-(const Vec4f& a, float s) { return a - Vec4f(s); }
inline Vec4f operator*(const Vec4f& a, float s) { return a * Vec4f(s); }
inline Vec4f operator/(const Vec4f& a, float s) { return a / Vec4f(s); }
inline Vec4f operator+(float s, const Vec4f& a) { return a + s; }
inline Vec4f operator-(float s, const Vec4f& a) { return Vec4f(s) - a; }
inline Vec4f operator*(float s, const Vec4f& a) { return a * s; }
inline Vec4f operator/(float s, const Vec4f& a) { return Vec4f(s) / a; }
inline Vec4f operator-(const Vec4f& a) { return Vec4f(simde_mm_sub_ps(simde_mm_setzero_ps(), a.v)); }
inline Vec4f& operator+=(Vec4f& a, const Vec4f& b) { a.v = simde_mm_add_ps(a.v, b.v); return a; }
inline Vec4f& operator-=(Vec4f& a, const Vec4f& b) { a.v = simde_mm_sub_ps(a.v, b.v); return a; }
inline Vec4f& operator*=(Vec4f& a, const Vec4f& b) { a.v = simde_mm_mul_ps(a.v, b.v); return a; }
inline Vec4f& operator/=(Vec4f& a, const Vec4f& b) { a.v = simde_mm_div_ps(a.v, b.v); return a; }
inline Vec4f& operator+=(Vec4f& a, float s) { a = a + s; return a; }
inline Vec4f& operator-=(Vec4f& a, float s) { a = a - s; return a; }
inline Vec4f& operator*=(Vec4f& a, float s) { a = a * s; return a; }
inline Vec4f& operator/=(Vec4f& a, float s) { a = a / s; return a; }

inline Vec8f operator+(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_add_ps(a.v, b.v)); }
inline Vec8f operator-(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_sub_ps(a.v, b.v)); }
inline Vec8f operator*(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_mul_ps(a.v, b.v)); }
inline Vec8f operator/(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_div_ps(a.v, b.v)); }
inline Vec8f operator+(const Vec8f& a, float s) { return a + Vec8f(s); }
inline Vec8f operator-(const Vec8f& a, float s) { return a - Vec8f(s); }
inline Vec8f operator*(const Vec8f& a, float s) { return a * Vec8f(s); }
inline Vec8f operator/(const Vec8f& a, float s) { return a / Vec8f(s); }
inline Vec8f operator+(float s, const Vec8f& a) { return a + s; }
inline Vec8f operator-(float s, const Vec8f& a) { return Vec8f(s) - a; }
inline Vec8f operator*(float s, const Vec8f& a) { return a * s; }
inline Vec8f operator/(float s, const Vec8f& a) { return Vec8f(s) / a; }
inline Vec8f operator-(const Vec8f& a) { return Vec8f(simde_mm256_sub_ps(simde_mm256_setzero_ps(), a.v)); }
inline Vec8f& operator+=(Vec8f& a, const Vec8f& b) { a.v = simde_mm256_add_ps(a.v, b.v); return a; }
inline Vec8f& operator-=(Vec8f& a, const Vec8f& b) { a.v = simde_mm256_sub_ps(a.v, b.v); return a; }
inline Vec8f& operator*=(Vec8f& a, const Vec8f& b) { a.v = simde_mm256_mul_ps(a.v, b.v); return a; }
inline Vec8f& operator/=(Vec8f& a, const Vec8f& b) { a.v = simde_mm256_div_ps(a.v, b.v); return a; }
inline Vec8f& operator+=(Vec8f& a, float s) { a = a + s; return a; }
inline Vec8f& operator-=(Vec8f& a, float s) { a = a - s; return a; }
inline Vec8f& operator*=(Vec8f& a, float s) { a = a * s; return a; }
inline Vec8f& operator/=(Vec8f& a, float s) { a = a / s; return a; }

inline Vec4d operator+(const Vec4d& a, const Vec4d& b) { return Vec4d(simde_mm256_add_pd(a.v, b.v)); }
inline Vec4d operator-(const Vec4d& a, const Vec4d& b) { return Vec4d(simde_mm256_sub_pd(a.v, b.v)); }
inline Vec4d operator*(const Vec4d& a, const Vec4d& b) { return Vec4d(simde_mm256_mul_pd(a.v, b.v)); }
inline Vec4d operator/(const Vec4d& a, const Vec4d& b) { return Vec4d(simde_mm256_div_pd(a.v, b.v)); }
inline Vec4d operator-(const Vec4d& a) { return Vec4d(simde_mm256_sub_pd(simde_mm256_setzero_pd(), a.v)); }
inline Vec4d operator+(const Vec4d& a, double s) { return a + Vec4d(s); }
inline Vec4d operator-(const Vec4d& a, double s) { return a - Vec4d(s); }
inline Vec4d operator*(const Vec4d& a, double s) { return a * Vec4d(s); }
inline Vec4d operator/(const Vec4d& a, double s) { return a / Vec4d(s); }
inline Vec4d operator+(double s, const Vec4d& a) { return a + s; }
inline Vec4d operator-(double s, const Vec4d& a) { return Vec4d(s) - a; }
inline Vec4d operator*(double s, const Vec4d& a) { return a * s; }
inline Vec4d operator/(double s, const Vec4d& a) { return Vec4d(s) / a; }
inline Vec4d& operator+=(Vec4d& a, const Vec4d& b) { a.v = simde_mm256_add_pd(a.v, b.v); return a; }
inline Vec4d& operator-=(Vec4d& a, const Vec4d& b) { a.v = simde_mm256_sub_pd(a.v, b.v); return a; }
inline Vec4d& operator*=(Vec4d& a, const Vec4d& b) { a.v = simde_mm256_mul_pd(a.v, b.v); return a; }
inline Vec4d& operator/=(Vec4d& a, const Vec4d& b) { a.v = simde_mm256_div_pd(a.v, b.v); return a; }
inline Vec4d& operator+=(Vec4d& a, double s) { a = a + s; return a; }
inline Vec4d& operator-=(Vec4d& a, double s) { a = a - s; return a; }
inline Vec4d& operator*=(Vec4d& a, double s) { a = a * s; return a; }
inline Vec4d& operator/=(Vec4d& a, double s) { a = a / s; return a; }

///////////////////////////////////////////////////////////////////////////////
// Arithmetic helpers (int)
///////////////////////////////////////////////////////////////////////////////
inline Vec4i operator+(const Vec4i& a, const Vec4i& b) { return Vec4i(simde_mm_add_epi32(a.v, b.v)); }
inline Vec4i operator-(const Vec4i& a, const Vec4i& b) { return Vec4i(simde_mm_sub_epi32(a.v, b.v)); }
inline Vec4i operator*(const Vec4i& a, const Vec4i& b) { return Vec4i(simde_mm_mullo_epi32(a.v, b.v)); }
inline Vec4i operator-(const Vec4i& a) { return Vec4i(simde_mm_sub_epi32(simde_mm_setzero_si128(), a.v)); }
inline Vec4i operator+(const Vec4i& a, int32_t s) { return a + Vec4i(s); }
inline Vec4i operator-(const Vec4i& a, int32_t s) { return a - Vec4i(s); }
inline Vec4i operator*(const Vec4i& a, int32_t s) { return a * Vec4i(s); }
inline Vec4i operator*(int32_t s, const Vec4i& a) { return a * s; }
inline Vec4i operator-(int32_t s, const Vec4i& a) { return Vec4i(s) - a; }
inline Vec4i operator>>(const Vec4i& a, int s) { return Vec4i(simde_mm_srli_epi32(a.v, s)); }
inline Vec4i operator<<(const Vec4i& a, int s) { return Vec4i(simde_mm_slli_epi32(a.v, s)); }
inline Vec4i& operator+=(Vec4i& a, const Vec4i& b) { a.v = simde_mm_add_epi32(a.v, b.v); return a; }
inline Vec4i& operator-=(Vec4i& a, const Vec4i& b) { a.v = simde_mm_sub_epi32(a.v, b.v); return a; }
inline Vec4i& operator*=(Vec4i& a, const Vec4i& b) { a.v = simde_mm_mullo_epi32(a.v, b.v); return a; }
inline Vec4i& operator+=(Vec4i& a, int32_t s) { a = a + s; return a; }
inline Vec4i& operator-=(Vec4i& a, int32_t s) { a = a - s; return a; }
inline Vec4i& operator*=(Vec4i& a, int32_t s) { a = a * s; return a; }
inline Vec4i& operator>>=(Vec4i& a, int s) { a.v = simde_mm_srli_epi32(a.v, s); return a; }
inline Vec4i operator&(const Vec4i& a, const Vec4i& b) { return Vec4i(simde_mm_and_si128(a.v, b.v)); }
inline Vec4i operator|(const Vec4i& a, const Vec4i& b) { return Vec4i(simde_mm_or_si128(a.v, b.v)); }
inline Vec4i operator^(const Vec4i& a, const Vec4i& b) { return Vec4i(simde_mm_xor_si128(a.v, b.v)); }
inline Vec4i operator~(const Vec4i& a) { return Vec4i(simde_mm_xor_si128(a.v, simde_mm_set1_epi32(-1))); }
inline Vec4i operator&(const Vec4i& a, int32_t s) { return a & Vec4i(s); }
inline Vec4i operator|(const Vec4i& a, int32_t s) { return a | Vec4i(s); }
inline Vec4i operator^(const Vec4i& a, int32_t s) { return a ^ Vec4i(s); }
inline Vec4i& operator&=(Vec4i& a, const Vec4i& b) { a.v = simde_mm_and_si128(a.v, b.v); return a; }
inline Vec4i& operator|=(Vec4i& a, const Vec4i& b) { a.v = simde_mm_or_si128(a.v, b.v); return a; }
inline Vec4i& operator^=(Vec4i& a, const Vec4i& b) { a.v = simde_mm_xor_si128(a.v, b.v); return a; }
inline Vec4i operator&&(const Vec4i& a, const Vec4i& b) { return a & b; }
inline Vec4i operator||(const Vec4i& a, const Vec4i& b) { return a | b; }

inline Vec8i operator+(const Vec8i& a, const Vec8i& b) { return Vec8i(simde_mm256_add_epi32(a.v, b.v)); }
inline Vec8i operator-(const Vec8i& a, const Vec8i& b) { return Vec8i(simde_mm256_sub_epi32(a.v, b.v)); }
inline Vec8i operator*(const Vec8i& a, const Vec8i& b) { return Vec8i(simde_mm256_mullo_epi32(a.v, b.v)); }
inline Vec8i operator-(const Vec8i& a) { return Vec8i(simde_mm256_sub_epi32(simde_mm256_setzero_si256(), a.v)); }
inline Vec8i operator+(const Vec8i& a, int32_t s) { return a + Vec8i(s); }
inline Vec8i operator-(const Vec8i& a, int32_t s) { return a - Vec8i(s); }
inline Vec8i operator*(const Vec8i& a, int32_t s) { return a * Vec8i(s); }
inline Vec8i operator*(int32_t s, const Vec8i& a) { return a * s; }
inline Vec8i operator-(int32_t s, const Vec8i& a) { return Vec8i(s) - a; }
inline Vec8i operator>>(const Vec8i& a, int s) { return Vec8i(simde_mm256_srli_epi32(a.v, s)); }
inline Vec8i operator<<(const Vec8i& a, int s) { return Vec8i(simde_mm256_slli_epi32(a.v, s)); }
inline Vec8i& operator+=(Vec8i& a, const Vec8i& b) { a.v = simde_mm256_add_epi32(a.v, b.v); return a; }
inline Vec8i& operator-=(Vec8i& a, const Vec8i& b) { a.v = simde_mm256_sub_epi32(a.v, b.v); return a; }
inline Vec8i& operator*=(Vec8i& a, const Vec8i& b) { a.v = simde_mm256_mullo_epi32(a.v, b.v); return a; }
inline Vec8i& operator+=(Vec8i& a, int32_t s) { a = a + s; return a; }
inline Vec8i& operator-=(Vec8i& a, int32_t s) { a = a - s; return a; }
inline Vec8i& operator*=(Vec8i& a, int32_t s) { a = a * s; return a; }
inline Vec8i operator&(const Vec8i& a, const Vec8i& b) { return Vec8i(simde_mm256_and_si256(a.v, b.v)); }
inline Vec8i operator|(const Vec8i& a, const Vec8i& b) { return Vec8i(simde_mm256_or_si256(a.v, b.v)); }
inline Vec8i operator^(const Vec8i& a, const Vec8i& b) { return Vec8i(simde_mm256_xor_si256(a.v, b.v)); }
inline Vec8i operator~(const Vec8i& a) { return Vec8i(simde_mm256_xor_si256(a.v, simde_mm256_set1_epi32(-1))); }
inline Vec8i operator&(const Vec8i& a, int32_t s) { return a & Vec8i(s); }
inline Vec8i operator|(const Vec8i& a, int32_t s) { return a | Vec8i(s); }
inline Vec8i operator^(const Vec8i& a, int32_t s) { return a ^ Vec8i(s); }
inline Vec8i& operator&=(Vec8i& a, const Vec8i& b) { a.v = simde_mm256_and_si256(a.v, b.v); return a; }
inline Vec8i& operator|=(Vec8i& a, const Vec8i& b) { a.v = simde_mm256_or_si256(a.v, b.v); return a; }
inline Vec8i& operator^=(Vec8i& a, const Vec8i& b) { a.v = simde_mm256_xor_si256(a.v, b.v); return a; }
inline Vec8i operator&&(const Vec8i& a, const Vec8i& b) { return a & b; }
inline Vec8i operator||(const Vec8i& a, const Vec8i& b) { return a | b; }

inline Vec4uq operator&(const Vec4uq& a, const Vec4uq& b) { return Vec4uq(simde_mm256_and_si256(a.v, b.v)); }
inline Vec4uq operator|(const Vec4uq& a, const Vec4uq& b) { return Vec4uq(simde_mm256_or_si256(a.v, b.v)); }
inline Vec4uq operator^(const Vec4uq& a, const Vec4uq& b) { return Vec4uq(simde_mm256_xor_si256(a.v, b.v)); }
inline Vec4uq operator~(const Vec4uq& a) { return Vec4uq(simde_mm256_xor_si256(a.v, simde_mm256_set1_epi64x(-1))); }
inline Vec4uq operator<<(const Vec4uq& a, int s) { return Vec4uq(simde_mm256_slli_epi64(a.v, s)); }
inline Vec4uq operator>>(const Vec4uq& a, int s) { return Vec4uq(simde_mm256_srli_epi64(a.v, s)); }
inline Vec4uq& operator&=(Vec4uq& a, const Vec4uq& b) { a.v = simde_mm256_and_si256(a.v, b.v); return a; }
inline Vec4uq& operator|=(Vec4uq& a, const Vec4uq& b) { a.v = simde_mm256_or_si256(a.v, b.v); return a; }
inline Vec4uq& operator^=(Vec4uq& a, const Vec4uq& b) { a.v = simde_mm256_xor_si256(a.v, b.v); return a; }
inline Vec4uq operator&(const Vec4uq& a, uint64_t s) { return a & Vec4uq(s); }
inline Vec4uq operator|(const Vec4uq& a, uint64_t s) { return Vec4uq(simde_mm256_or_si256(a.v, Vec4uq(s).v)); }
inline Vec4uq operator^(const Vec4uq& a, uint64_t s) { return Vec4uq(simde_mm256_xor_si256(a.v, Vec4uq(s).v)); }
inline Vec4uq operator&(uint64_t s, const Vec4uq& a) { return a & s; }
inline Vec4uq operator|(uint64_t s, const Vec4uq& a) { return a | s; }
inline Vec4uq operator^(uint64_t s, const Vec4uq& a) { return a ^ s; }

///////////////////////////////////////////////////////////////////////////////
// Comparisons
///////////////////////////////////////////////////////////////////////////////
inline Vec4f operator==(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_cmpeq_ps(a.v, b.v)); }
inline Vec4f operator!=(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_cmpneq_ps(a.v, b.v)); }
inline Vec4f operator<(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_cmplt_ps(a.v, b.v)); }
inline Vec4f operator>(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_cmpgt_ps(a.v, b.v)); }
inline Vec4f operator<=(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_cmple_ps(a.v, b.v)); }
inline Vec4f operator>=(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_cmpge_ps(a.v, b.v)); }
inline Vec4f operator==(const Vec4f& a, float s) { return a == Vec4f(s); }
inline Vec4f operator!=(const Vec4f& a, float s) { return a != Vec4f(s); }
inline Vec4f operator<(const Vec4f& a, float s) { return a < Vec4f(s); }
inline Vec4f operator>(const Vec4f& a, float s) { return a > Vec4f(s); }
inline Vec4f operator<=(const Vec4f& a, float s) { return a <= Vec4f(s); }
inline Vec4f operator>=(const Vec4f& a, float s) { return a >= Vec4f(s); }

inline Vec8f operator==(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_cmp_ps(a.v, b.v, SIMDE_CMP_EQ_OQ)); }
inline Vec8f operator!=(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_cmp_ps(a.v, b.v, SIMDE_CMP_NEQ_OQ)); }
inline Vec8f operator<(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_cmp_ps(a.v, b.v, SIMDE_CMP_LT_OQ)); }
inline Vec8f operator>(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_cmp_ps(a.v, b.v, SIMDE_CMP_GT_OQ)); }
inline Vec8f operator<=(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_cmp_ps(a.v, b.v, SIMDE_CMP_LE_OQ)); }
inline Vec8f operator>=(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_cmp_ps(a.v, b.v, SIMDE_CMP_GE_OQ)); }
inline Vec8f operator==(const Vec8f& a, float s) { return a == Vec8f(s); }
inline Vec8f operator!=(const Vec8f& a, float s) { return a != Vec8f(s); }
inline Vec8f operator<(const Vec8f& a, float s) { return a < Vec8f(s); }
inline Vec8f operator>(const Vec8f& a, float s) { return a > Vec8f(s); }
inline Vec8f operator<=(const Vec8f& a, float s) { return a <= Vec8f(s); }
inline Vec8f operator>=(const Vec8f& a, float s) { return a >= Vec8f(s); }

inline Vec4d operator==(const Vec4d& a, const Vec4d& b) { return Vec4d(simde_mm256_cmp_pd(a.v, b.v, SIMDE_CMP_EQ_OQ)); }
inline Vec4d operator!=(const Vec4d& a, const Vec4d& b) { return Vec4d(simde_mm256_cmp_pd(a.v, b.v, SIMDE_CMP_NEQ_OQ)); }
inline Vec4d operator<(const Vec4d& a, const Vec4d& b) { return Vec4d(simde_mm256_cmp_pd(a.v, b.v, SIMDE_CMP_LT_OQ)); }
inline Vec4d operator>(const Vec4d& a, const Vec4d& b) { return Vec4d(simde_mm256_cmp_pd(a.v, b.v, SIMDE_CMP_GT_OQ)); }
inline Vec4d operator<=(const Vec4d& a, const Vec4d& b) { return Vec4d(simde_mm256_cmp_pd(a.v, b.v, SIMDE_CMP_LE_OQ)); }
inline Vec4d operator>=(const Vec4d& a, const Vec4d& b) { return Vec4d(simde_mm256_cmp_pd(a.v, b.v, SIMDE_CMP_GE_OQ)); }

inline Vec4i operator==(const Vec4i& a, const Vec4i& b) { return Vec4i(simde_mm_cmpeq_epi32(a.v, b.v)); }
inline Vec4i operator!=(const Vec4i& a, const Vec4i& b) { return ~ (a == b); }
inline Vec4i operator>(const Vec4i& a, const Vec4i& b) { return Vec4i(simde_mm_cmpgt_epi32(a.v, b.v)); }
inline Vec4i operator<(const Vec4i& a, const Vec4i& b) { return Vec4i(simde_mm_cmpgt_epi32(b.v, a.v)); }
inline Vec4i operator>=(const Vec4i& a, const Vec4i& b) { return ~ (a < b); }
inline Vec4i operator<=(const Vec4i& a, const Vec4i& b) { return ~ (a > b); }
inline Vec4i operator==(const Vec4i& a, int32_t s) { return a == Vec4i(s); }
inline Vec4i operator!=(const Vec4i& a, int32_t s) { return a != Vec4i(s); }
inline Vec4i operator>(const Vec4i& a, int32_t s) { return a > Vec4i(s); }
inline Vec4i operator<(const Vec4i& a, int32_t s) { return a < Vec4i(s); }
inline Vec4i operator>=(const Vec4i& a, int32_t s) { return a >= Vec4i(s); }
inline Vec4i operator<=(const Vec4i& a, int32_t s) { return a <= Vec4i(s); }

inline Vec8i operator==(const Vec8i& a, const Vec8i& b) { return Vec8i(simde_mm256_cmpeq_epi32(a.v, b.v)); }
inline Vec8i operator!=(const Vec8i& a, const Vec8i& b) { return ~ (a == b); }
inline Vec8i operator>(const Vec8i& a, const Vec8i& b) { return Vec8i(simde_mm256_cmpgt_epi32(a.v, b.v)); }
inline Vec8i operator<(const Vec8i& a, const Vec8i& b) { return Vec8i(simde_mm256_cmpgt_epi32(b.v, a.v)); }
inline Vec8i operator>=(const Vec8i& a, const Vec8i& b) { return ~ (a < b); }
inline Vec8i operator<=(const Vec8i& a, const Vec8i& b) { return ~ (a > b); }
inline Vec8i operator==(const Vec8i& a, int32_t s) { return a == Vec8i(s); }
inline Vec8i operator!=(const Vec8i& a, int32_t s) { return a != Vec8i(s); }
inline Vec8i operator>(const Vec8i& a, int32_t s) { return a > Vec8i(s); }
inline Vec8i operator<(const Vec8i& a, int32_t s) { return a < Vec8i(s); }
inline Vec8i operator>=(const Vec8i& a, int32_t s) { return a >= Vec8i(s); }
inline Vec8i operator<=(const Vec8i& a, int32_t s) { return a <= Vec8i(s); }

///////////////////////////////////////////////////////////////////////////////
// Min/Max/Abs
///////////////////////////////////////////////////////////////////////////////
inline Vec4f max(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_max_ps(a.v, b.v)); }
inline Vec4f min(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_min_ps(a.v, b.v)); }
inline Vec8f max(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_max_ps(a.v, b.v)); }
inline Vec8f min(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_min_ps(a.v, b.v)); }
inline Vec4f max(const Vec4f& a, float s) { return max(a, Vec4f(s)); }
inline Vec4f min(const Vec4f& a, float s) { return min(a, Vec4f(s)); }
inline Vec8f max(const Vec8f& a, float s) { return max(a, Vec8f(s)); }
inline Vec8f min(const Vec8f& a, float s) { return min(a, Vec8f(s)); }
inline Vec4f max(float s, const Vec4f& a) { return max(Vec4f(s), a); }
inline Vec4f min(float s, const Vec4f& a) { return min(Vec4f(s), a); }
inline Vec8f max(float s, const Vec8f& a) { return max(Vec8f(s), a); }
inline Vec8f min(float s, const Vec8f& a) { return min(Vec8f(s), a); }
inline Vec4f abs(const Vec4f& a) { return Vec4f(simde_mm_and_ps(a.v, simde_mm_castsi128_ps(simde_mm_set1_epi32(0x7fffffff)))); }
inline Vec8f abs(const Vec8f& a) { return Vec8f(simde_mm256_and_ps(a.v, simde_mm256_castsi256_ps(simde_mm256_set1_epi32(0x7fffffff)))); }

inline Vec4i max(const Vec4i& a, const Vec4i& b) { return Vec4i(simde_mm_max_epi32(a.v, b.v)); }
inline Vec4i min(const Vec4i& a, const Vec4i& b) { return Vec4i(simde_mm_min_epi32(a.v, b.v)); }
inline Vec8i max(const Vec8i& a, const Vec8i& b) { return Vec8i(simde_mm256_max_epi32(a.v, b.v)); }
inline Vec8i min(const Vec8i& a, const Vec8i& b) { return Vec8i(simde_mm256_min_epi32(a.v, b.v)); }
inline Vec4i max(const Vec4i& a, int s) { return max(a, Vec4i(s)); }
inline Vec4i max(int s, const Vec4i& a) { return max(Vec4i(s), a); }
inline Vec4i min(const Vec4i& a, int s) { return min(a, Vec4i(s)); }
inline Vec4i min(int s, const Vec4i& a) { return min(Vec4i(s), a); }

///////////////////////////////////////////////////////////////////////////////
// Fused multiply add
///////////////////////////////////////////////////////////////////////////////
inline Vec4f fmadd(const Vec4f& a, const Vec4f& b, const Vec4f& c) { return Vec4f(simde_mm_fmadd_ps(a.v, b.v, c.v)); }
inline Vec8f fmadd(const Vec8f& a, const Vec8f& b, const Vec8f& c) { return Vec8f(simde_mm256_fmadd_ps(a.v, b.v, c.v)); }
inline Vec4f mul_add(const Vec4f& a, const Vec4f& b, const Vec4f& c) { return fmadd(a, b, c); }
inline Vec8f mul_add(const Vec8f& a, const Vec8f& b, const Vec8f& c) { return fmadd(a, b, c); }

///////////////////////////////////////////////////////////////////////////////
// Select / masks
///////////////////////////////////////////////////////////////////////////////
inline Vec4f operator&(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_and_ps(a.v, b.v)); }
inline Vec4f operator|(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_or_ps(a.v, b.v)); }
inline Vec4f operator^(const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_xor_ps(a.v, b.v)); }
inline Vec4f operator||(const Vec4f& a, const Vec4f& b) { return a | b; }
inline Vec4f operator&&(const Vec4f& a, const Vec4f& b) { return a & b; }

inline Vec8f operator&(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_and_ps(a.v, b.v)); }
inline Vec8f operator|(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_or_ps(a.v, b.v)); }
inline Vec8f operator^(const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_xor_ps(a.v, b.v)); }
inline Vec8f operator||(const Vec8f& a, const Vec8f& b) { return a | b; }
inline Vec8f operator&&(const Vec8f& a, const Vec8f& b) { return a & b; }

inline Vec4f select(const Vec4f& m, const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_blendv_ps(b.v, a.v, m.v)); }
inline Vec4f select(const Vec4i& m, const Vec4f& a, const Vec4f& b) { return Vec4f(simde_mm_blendv_ps(b.v, a.v, simde_mm_castsi128_ps(m.v))); }
inline Vec4i select(const Vec4i& m, const Vec4i& a, const Vec4i& b) { return Vec4i(simde_mm_blendv_epi8(b.v, a.v, m.v)); }
inline Vec4f select(const Vec4f& m, const Vec4f& a, float b) { return select(m, a, Vec4f(b)); }
inline Vec4f select(const Vec4f& m, float a, const Vec4f& b) { return select(m, Vec4f(a), b); }
inline Vec4f select(const Vec4i& m, const Vec4f& a, float b) { return select(m, a, Vec4f(b)); }
inline Vec4f select(const Vec4i& m, float a, const Vec4f& b) { return select(m, Vec4f(a), b); }
inline Vec4i select(const Vec4i& m, const Vec4i& a, int b) { return select(m, a, Vec4i(b)); }
inline Vec4i select(const Vec4i& m, int a, const Vec4i& b) { return select(m, Vec4i(a), b); }

inline Vec8f select(const Vec8f& m, const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_blendv_ps(b.v, a.v, m.v)); }
inline Vec8f select(const Vec8i& m, const Vec8f& a, const Vec8f& b) { return Vec8f(simde_mm256_blendv_ps(b.v, a.v, simde_mm256_castsi256_ps(m.v))); }
inline Vec8i select(const Vec8i& m, const Vec8i& a, const Vec8i& b) { return Vec8i(simde_mm256_blendv_epi8(b.v, a.v, m.v)); }
inline Vec8f select(const Vec8f& m, const Vec8f& a, float b) { return select(m, a, Vec8f(b)); }
inline Vec8f select(const Vec8f& m, float a, const Vec8f& b) { return select(m, Vec8f(a), b); }
inline Vec8f select(const Vec8i& m, const Vec8f& a, float b) { return select(m, a, Vec8f(b)); }
inline Vec8f select(const Vec8i& m, float a, const Vec8f& b) { return select(m, Vec8f(a), b); }
inline Vec8i select(const Vec8i& m, const Vec8i& a, int b) { return select(m, a, Vec8i(b)); }
inline Vec8i select(const Vec8i& m, int a, const Vec8i& b) { return select(m, Vec8i(a), b); }

inline bool any(const Vec4f& m) { return simde_mm_movemask_ps(m.v) != 0; }
inline bool any(const Vec4i& m) { return simde_mm_movemask_epi8(m.v) != 0; }
inline bool any(const Vec8f& m) { return simde_mm256_movemask_ps(m.v) != 0; }
inline bool any(const Vec8i& m) { return simde_mm256_movemask_epi8(m.v) != 0; }
inline bool vany3(const Vec4i& m) { return (simde_mm_movemask_ps(simde_mm_castsi128_ps(m.v)) & 7) != 0; }

inline bool vall(const Vec4f& m) { return simde_mm_movemask_ps(m.v) == 0xF; }
inline bool vall(const Vec4i& m) { return simde_mm_movemask_epi8(m.v) == 0xFFFF; }
inline bool vall(const Vec8f& m) { return simde_mm256_movemask_ps(m.v) == 0xFF; }
inline bool vall(const Vec8i& m) { return simde_mm256_movemask_epi8(m.v) == -1; }
inline bool vall3(const Vec4i& m) { return (simde_mm_movemask_ps(simde_mm_castsi128_ps(m.v)) & 7) == 7; }

inline bool vany(const Vec4f& m) { return any(m); }
inline bool vany(const Vec4i& m) { return any(m); }
inline bool vany(const Vec8f& m) { return any(m); }
inline bool vany(const Vec8i& m) { return any(m); }

///////////////////////////////////////////////////////////////////////////////
// Math helpers
///////////////////////////////////////////////////////////////////////////////
inline Vec4f floor(const Vec4f& a) { return Vec4f(simde_mm_floor_ps(a.v)); }
inline Vec8f floor(const Vec8f& a) { return Vec8f(simde_mm256_floor_ps(a.v)); }
inline Vec4d floor(const Vec4d& a) { return Vec4d(simde_mm256_floor_pd(a.v)); }
inline Vec4f ceil(const Vec4f& a) { return Vec4f(simde_mm_ceil_ps(a.v)); }
inline Vec8f ceil(const Vec8f& a) { return Vec8f(simde_mm256_ceil_ps(a.v)); }
inline Vec4d ceil(const Vec4d& a) { return Vec4d(simde_mm256_ceil_pd(a.v)); }

inline Vec4f round(const Vec4f& a) { return Vec4f(simde_mm_round_ps(a.v, SIMDE_MM_FROUND_TO_NEAREST_INT | SIMDE_MM_FROUND_NO_EXC)); }
inline Vec8f round(const Vec8f& a) { return Vec8f(simde_mm256_round_ps(a.v, SIMDE_MM_FROUND_TO_NEAREST_INT | SIMDE_MM_FROUND_NO_EXC)); }
inline Vec4d round(const Vec4d& a) { return Vec4d(simde_mm256_round_pd(a.v, SIMDE_MM_FROUND_TO_NEAREST_INT | SIMDE_MM_FROUND_NO_EXC)); }

inline Vec4f sqrt(const Vec4f& a) { return Vec4f(simde_mm_sqrt_ps(a.v)); }
inline Vec8f sqrt(const Vec8f& a) { return Vec8f(simde_mm256_sqrt_ps(a.v)); }
inline Vec4f approx_rsqrt(const Vec4f& a) { return Vec4f(simde_mm_rsqrt_ps(a.v)); }
inline Vec8f approx_rsqrt(const Vec8f& a) { return Vec8f(simde_mm256_rsqrt_ps(a.v)); }

inline Vec4i truncate_to_int(const Vec4f& a) { return Vec4i(simde_mm_cvttps_epi32(a.v)); }
inline Vec8i truncate_to_int(const Vec8f& a) { return Vec8i(simde_mm256_cvttps_epi32(a.v)); }
inline Vec4i round_to_int(const Vec4f& a) { return Vec4i(simde_mm_cvtps_epi32(a.v)); }
inline Vec8i round_to_int(const Vec8f& a) { return Vec8i(simde_mm256_cvtps_epi32(a.v)); }
inline Vec4f to_float(const Vec4i& a) { return Vec4f(simde_mm_cvtepi32_ps(a.v)); }

inline Vec4f vfrac(const Vec4f& x) { return x - floor(x); }
inline Vec4f vlinstep(Vec4f a, Vec4f b, Vec4f x) { x = (x - a) / (b - a); return max(Vec4f(0.0f), min(x, Vec4f(1.0f))); }
inline Vec8f vlinstep(Vec8f a, Vec8f b, Vec8f x) { x = (x - a) / (b - a); return max(Vec8f(0.0f), min(x, Vec8f(1.0f))); }
inline Vec4f vsmoothstep(Vec4f a, Vec4f b, Vec4f x) {
  Vec4f y = (x - a) / (b - a);
  y = (y * y * (Vec4f(3.0f) - Vec4f(2.0f) * y));
  y = select(x < a, Vec4f(0.0f), select(x >= b, Vec4f(1.0f), y));
  return y;
}

inline Vec8f vpospart(const Vec8f& x) { return max(x, Vec8f(0.0f)); }
inline Vec8f vnegpart(const Vec8f& x) { return min(x, Vec8f(0.0f)); }

///////////////////////////////////////////////////////////////////////////////
// Finite checks
///////////////////////////////////////////////////////////////////////////////
inline bool visfinite(const Vec4f& a) {
  float tmp[4];
  a.store(tmp);
  return std::isfinite(tmp[0]) && std::isfinite(tmp[1]) && std::isfinite(tmp[2]) && std::isfinite(tmp[3]);
}
inline bool visfinite(const Vec8f& a) {
  float tmp[8];
  a.store(tmp);
  for (int i = 0; i < 8; ++i) if (!std::isfinite(tmp[i])) return false;
  return true;
}
inline bool visfinite(const Vec4d& a) {
  double tmp[4];
  a.store(tmp);
  return std::isfinite(tmp[0]) && std::isfinite(tmp[1]) && std::isfinite(tmp[2]) && std::isfinite(tmp[3]);
}

///////////////////////////////////////////////////////////////////////////////
// Lane access helpers
///////////////////////////////////////////////////////////////////////////////
inline int32_t vget_x(const Vec4i& vec) { return simde_mm_cvtsi128_si32(vec.v); }
inline int32_t vget_y(const Vec4i& vec) { return simde_mm_extract_epi32(vec.v, 1); }
inline int32_t vget_z(const Vec4i& vec) { return simde_mm_extract_epi32(vec.v, 2); }
inline int32_t vget_w(const Vec4i& vec) { return simde_mm_extract_epi32(vec.v, 3); }
inline void vset_w(Vec4i& vec, int x) { vec.v = simde_mm_insert_epi32(vec.v, x, 3); }

inline float vfget_x(const Vec4f& vec) { return simde_mm_cvtss_f32(vec.v); }
inline float vfget_y(const Vec4f& vec) { float tmp[4]; vec.store(tmp); return tmp[1]; }
inline float vfget_z(const Vec4f& vec) { float tmp[4]; vec.store(tmp); return tmp[2]; }
inline float vfget_w(const Vec4f& vec) { float tmp[4]; vec.store(tmp); return tmp[3]; }

inline double vfget_x(const Vec4d& vec) { double tmp[4]; vec.store(tmp); return tmp[0]; }
inline double vfget_y(const Vec4d& vec) { double tmp[4]; vec.store(tmp); return tmp[1]; }
inline double vfget_z(const Vec4d& vec) { double tmp[4]; vec.store(tmp); return tmp[2]; }
inline double vfget_w(const Vec4d& vec) { double tmp[4]; vec.store(tmp); return tmp[3]; }

///////////////////////////////////////////////////////////////////////////////
// Conversions
///////////////////////////////////////////////////////////////////////////////
inline Vec4f vint2float(Vec4i x) { return Vec4f(simde_mm_cvtepi32_ps(x.v)); }
inline Vec4f vuint2float(Vec4ui x) { return Vec4f(simde_mm_cvtepi32_ps(x.v)); }
inline Vec8f vuint2float(Vec8ui x) { return Vec8f(simde_mm256_cvtepi32_ps(x.v)); }

inline Vec4f Vec4f_zero() { return Vec4f(simde_mm_setzero_ps()); }
inline Vec4i Vec4i_zero() { return Vec4i(simde_mm_setzero_si128()); }

///////////////////////////////////////////////////////////////////////////////
// Shuffles and lerp
///////////////////////////////////////////////////////////////////////////////
enum VectorSelect {
  Ax = 0, Ay = 1, Az = 2, Aw = 3,
  Bx = 8, By = 9, Bz = 10, Bw = 11,
};

template <VectorSelect S0, VectorSelect S1, VectorSelect S2, VectorSelect S3>
inline Vec4f vshuffle_ps(Vec4f x, Vec4f y) {
  int imm = S0 + S1 * 4 + (S2 - Bx) * 16 + (S3 - Bx) * 64;
  return Vec4f(simde_mm_shuffle_ps(x.v, y.v, imm));
}

template <VectorSelect S0, VectorSelect S1, VectorSelect S2, VectorSelect S3>
inline Vec8f vshuffle_ps(Vec8f x, Vec8f y) {
  Vec4f lo = vshuffle_ps<S0, S1, S2, S3>(x.get_low(), y.get_low());
  Vec4f hi = vshuffle_ps<S0, S1, S2, S3>(x.get_high(), y.get_high());
  return Vec8f(lo, hi);
}

inline Vec4f vlerp_ps(const Vec4f& t, const Vec4f& a, const Vec4f& b) { return mul_add(b - a, t, a); }
inline Vec8f vlerp_ps(const Vec8f& t, const Vec8f& a, const Vec8f& b) { return mul_add(b - a, t, a); }

///////////////////////////////////////////////////////////////////////////////
// Permute helpers for 8-wide vectors
///////////////////////////////////////////////////////////////////////////////
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline Vec8f permute8(const Vec8f& v) {
  float tmp[8];
  v.store(tmp);
  float out[8] = {tmp[i0], tmp[i1], tmp[i2], tmp[i3], tmp[i4], tmp[i5], tmp[i6], tmp[i7]};
  Vec8f r;
  r.load(out);
  return r;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline Vec8i permute8(const Vec8i& v) {
  int32_t tmp[8];
  v.store(tmp);
  int32_t out[8] = {tmp[i0], tmp[i1], tmp[i2], tmp[i3], tmp[i4], tmp[i5], tmp[i6], tmp[i7]};
  Vec8i r;
  r.load(out);
  return r;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline Vec8f blend8f(const Vec8f& a, const Vec8f& b) {
  float av[8], bv[8], out[8];
  a.store(av);
  b.store(bv);
  int idx[8] = {i0, i1, i2, i3, i4, i5, i6, i7};
  for (int k = 0; k < 8; ++k) {
    int s = idx[k];
    out[k] = (s < 8) ? av[s] : bv[s - 8];
  }
  Vec8f r;
  r.load(out);
  return r;
}

///////////////////////////////////////////////////////////////////////////////
// Horizontal reductions
///////////////////////////////////////////////////////////////////////////////
inline float horizontal_sum(const Vec4f& x) { return v4f_hsum(x.v); }
inline float horizontal_sum(const Vec8f& x) { return v8f_hsum(x.v); }
inline float horizontal_add(const Vec4f& x) { return horizontal_sum(x); }
inline float horizontal_add(const Vec8f& x) { return horizontal_sum(x); }
inline double horizontal_add(const Vec4d& x) {
  double tmp[4];
  x.store(tmp);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}
inline int horizontal_add(const Vec4i& x) {
  alignas(16) int32_t tmp[4];
  x.store(tmp);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

inline float horizontal_max(const Vec4f& x) {
  float tmp[4];
  x.store(tmp);
  return std::max(std::max(tmp[0], tmp[1]), std::max(tmp[2], tmp[3]));
}

inline float horizontal_max(const Vec4d& a) {
  double tmp[4];
  a.store(tmp);
  return static_cast<float>(std::max(std::max(tmp[0], tmp[1]), std::max(tmp[2], tmp[3])));
}
inline float horizontal_max(const Vec8f& x) {
  float tmp[8];
  x.store(tmp);
  float m = tmp[0];
  for (int i = 1; i < 8; ++i) m = std::max(m, tmp[i]);
  return m;
}

///////////////////////////////////////////////////////////////////////////////
// Masked store
///////////////////////////////////////////////////////////////////////////////
inline void vmaskstore(float* dest, const Vec4f& x, const Vec4i& mask) {
  alignas(16) float tmp[4];
  alignas(16) int32_t m[4];
  x.store(tmp);
  simde_mm_store_si128(reinterpret_cast<simde__m128i*>(m), mask.v);
  for (int i = 0; i < 4; ++i) if (m[i]) dest[i] = tmp[i];
}

inline void vmaskstore(float* dest, const Vec8f& x, const Vec8i& mask) {
  alignas(32) float tmp[8];
  alignas(32) int32_t m[8];
  x.store(tmp);
  mask.store(m);
  for (int i = 0; i < 8; ++i) if (m[i]) dest[i] = tmp[i];
}

inline void vmaskstore4(float* dest, const Vec4f& x, const Vec4i& mask) { vmaskstore(dest, x, mask); }
inline void vmaskstore4(double* dest, const Vec4d& x, const Vec4i& mask) {
  alignas(32) double tmp[4];
  alignas(16) int32_t m[4];
  x.store(tmp);
  simde_mm_store_si128(reinterpret_cast<simde__m128i*>(m), mask.v);
  for (int i = 0; i < 4; ++i) if (m[i]) dest[i] = tmp[i];
}

inline Vec4f vmask4(const Vec4f& x, const Vec4i& mask) { return Vec4f(simde_mm_and_ps(x.v, simde_mm_castsi128_ps(mask.v))); }
inline Vec4d vmask4(const Vec4d& x, const Vec4i& mask) {
  simde__m256i mask64 = simde_mm256_cvtepi32_epi64(mask.v);
  return Vec4d(simde_mm256_and_pd(x.v, simde_mm256_castsi256_pd(mask64)));
}

inline int horizontal_or(const Vec4i& m) {
  alignas(16) int32_t vals[4];
  m.store(vals);
  return vals[0] | vals[1] | vals[2] | vals[3];
}
inline int horizontal_or(const Vec4f& m) { return any(m); }
inline int horizontal_or(const Vec8i& m) {
  alignas(32) int32_t vals[8];
  m.store(vals);
  int r = 0;
  for (int i = 0; i < 8; ++i) r |= vals[i];
  return r;
}
inline int horizontal_or(const Vec8f& m) { return any(m); }

///////////////////////////////////////////////////////////////////////////////
// Load/store helpers for byte/short data
///////////////////////////////////////////////////////////////////////////////
inline Vec8us compress(const Vec4ui& lo, const Vec4ui& hi) { return Vec8us(simde_mm_packus_epi32(lo.v, hi.v)); }

inline void vload_from_uint16_a(Vec8ui& z, const uint16_t* p) {
  Vec8us y = Vec8us().load_a(p);
  Vec4ui y4l = Vec4ui(simde_mm_unpacklo_epi16(y.v, simde_mm_setzero_si128()));
  Vec4ui y4h = Vec4ui(simde_mm_unpackhi_epi16(y.v, simde_mm_setzero_si128()));
  z = Vec8ui(y4l, y4h);
}

inline void vload_from_uint16_a(Vec4ui& z, const uint16_t* p) {
  z = Vec4ui(simde_mm_unpacklo_epi16(Vec8us(p[0], p[1], p[2], p[3], 0, 0, 0, 0).v, simde_mm_setzero_si128()));
}

inline void vload_from_uint16_a(Vec8f& w, const uint16_t* p) {
  Vec8ui t;
  vload_from_uint16_a(t, p);
  w = vuint2float(t);
}

inline void vload_from_uint16_a(Vec4f& z, const uint16_t* p) {
  Vec4ui w16 = Vec4ui(simde_mm_unpacklo_epi16(Vec8us(p[0], p[1], p[2], p[3], 0, 0, 0, 0).v, simde_mm_setzero_si128()));
  z = vint2float(w16);
}

inline void vload_from_uint8_a(Vec8ui& w, uint8_t* source) {
  uint8_t tmp[8];
  std::memcpy(tmp, source, 8);
  uint32_t vals[8];
  for (int i = 0; i < 8; ++i) vals[i] = tmp[i];
  w.load(vals);
}

inline void vload_from_uint8_a(Vec8f& w, uint8_t* source) {
  float vals[8];
  for (int i = 0; i < 8; ++i) vals[i] = static_cast<float>(source[i]);
  w.load(vals);
}

inline void vload_from_uint8_a(Vec4f& z, const uint8_t* p) { z = Vec4f(p[0], p[1], p[2], p[3]); }

inline void vload8(Vec8ui& w, const uint8_t* source) { vload_from_uint8_a(w, const_cast<uint8_t*>(source)); }
inline void vload8(Vec8ui& w, const uint32_t* source) { w.load(source); }

inline void vstore_to_uint16_a(const Vec8f& x, uint16_t* p) {
  Vec8i xi = truncate_to_int(round(x));
  Vec8us packed = compress(xi.get_low(), xi.get_high());
  packed.store_a(p);
}

inline void vstore_to_uint16_a(const Vec4f& x, uint16_t* p) {
  Vec4i xi = truncate_to_int(round(x));
  p[0] = static_cast<uint16_t>(vget_x(xi));
  p[1] = static_cast<uint16_t>(vget_y(xi));
  p[2] = static_cast<uint16_t>(vget_z(xi));
  p[3] = static_cast<uint16_t>(vget_w(xi));
}

inline void vstore_to_uint16_a(const Vec8ui& xi, uint16_t* p) {
  Vec8us xus = compress(xi.get_low(), xi.get_high());
  xus.store_a(p);
}

inline void vstore_to_uint16_a(const Vec4ui& xi, uint16_t* p) {
  p[0] = static_cast<uint16_t>(vget_x(xi));
  p[1] = static_cast<uint16_t>(vget_y(xi));
  p[2] = static_cast<uint16_t>(vget_z(xi));
  p[3] = static_cast<uint16_t>(vget_w(xi));
}

inline void vstore_to_uint16_nt(const Vec4f& x, uint16_t* p) { vstore_to_uint16_a(x, p); }
inline void vstore_to_uint16_nt(const Vec8ui& xi, uint16_t* p) { vstore_to_uint16_a(xi, p); }
inline void vstore_to_uint16_nt(const Vec4ui& xi, uint16_t* p) { vstore_to_uint16_a(xi, p); }

inline void vstore_to_uint8_a(const Vec8ui& xi, uint8_t* p) {
  alignas(32) uint32_t tmp[8];
  xi.store(tmp);
  for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(tmp[i]);
}

inline void vstore8(Vec8ui& w, uint32_t* dest) { w.store(dest); }

///////////////////////////////////////////////////////////////////////////////
// Gather
///////////////////////////////////////////////////////////////////////////////
inline Vec4f vgather4(const float* base, const Vec4i& indices) {
  alignas(16) int32_t idx[4];
  indices.store(idx);
  float vals[4] = {base[idx[0]], base[idx[1]], base[idx[2]], base[idx[3]]};
  Vec4f r;
  r.load(vals);
  return r;
}

inline Vec4f vgather4(const float** bases, const Vec4i& indices, const int n) {
  alignas(16) int32_t idx[4];
  indices.store(idx);
  float vals[4] = {0.f, 0.f, 0.f, 0.f};
  for (int i = 0; i < n && i < 4; ++i) vals[i] = bases[i][idx[i]];
  Vec4f r;
  r.load(vals);
  return r;
}

inline Vec8f vgather8(const float* base, const Vec8i& indices) {
  alignas(32) int32_t idx[8];
  indices.store(idx);
  float vals[8];
  for (int i = 0; i < 8; ++i) vals[i] = base[idx[i]];
  Vec8f r;
  r.load(vals);
  return r;
}

inline Vec8f vgather8(const float** bases, const Vec8i& indices, const int n) {
  alignas(32) int32_t idx[8];
  indices.store(idx);
  float vals[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int i = 0; i < n && i < 8; ++i) vals[i] = bases[i][idx[i]];
  Vec8f r;
  r.load(vals);
  return r;
}

inline Vec8f vgather8(const float** bases, const int64_t idx, const int n) {
  float vals[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int i = 0; i < n && i < 8; ++i) vals[i] = bases[i][idx];
  Vec8f r;
  r.load(vals);
  return r;
}

///////////////////////////////////////////////////////////////////////////////
// Streaming load/store (mapped to regular load/store on ARM)
///////////////////////////////////////////////////////////////////////////////
inline void vstream(float* p, Vec4f ymm) { ymm.store(p); }
inline void vstream(float* p, Vec8f ymm) { ymm.store(p); }
inline void vstream(double* p, Vec4d ymm) { ymm.store(p); }

///////////////////////////////////////////////////////////////////////////////
// Misc helpers
///////////////////////////////////////////////////////////////////////////////
template<int pos>
inline Vec4f v4f_zero_to(const Vec4f& v) {
  float tmp[4];
  v.store(tmp);
  for (int i = pos; i < 4; ++i) tmp[i] = 0.0f;
  Vec4f r;
  r.load(tmp);
  return r;
}

inline float v3normSq(const Vec4f& vec) {
  Vec4f v = v4f_zero_to<3>(vec);
  v *= v;
  return vfget_x(v) + vfget_y(v) + vfget_z(v);
}

inline Vec4i vzero() { return Vec4i_zero(); }
inline Vec4f vfzero() { return Vec4f_zero(); }
inline float vmax(float a, float b) { return std::max(a, b); }
inline float vmin(float a, float b) { return std::min(a, b); }

///////////////////////////////////////////////////////////////////////////////
// Double/float conversions
///////////////////////////////////////////////////////////////////////////////
inline Vec4f vconv_pd_ps(const Vec4d& a) { return Vec4f(simde_mm256_cvtpd_ps(a.v)); }
inline Vec4d vconv_ps_pd(const Vec4f& a) { return Vec4d(simde_mm256_cvtps_pd(a.v)); }

inline Vec4d loadf(const float* p) { return Vec4d(simde_mm256_cvtps_pd(simde_mm_loadu_ps(p))); }
inline Vec4d loadf_a(const float* p) { return Vec4d(simde_mm256_cvtps_pd(simde_mm_load_ps(p))); }
inline Vec4d loadf_a(const double* p) { return Vec4d().load_a(p); }

inline void vstreamf_a(float* p, const Vec4d& v) {
  Vec4f vf = vconv_pd_ps(v);
  vf.store_a(p);
}
inline void vstreamf_a(double* p, const Vec4d& v) { v.store_a(p); }
inline void vstoref_a(float* p, const Vec4d& v) { vstreamf_a(p, v); }
inline void vstoref_a(double* p, const Vec4d& v) { v.store_a(p); }

///////////////////////////////////////////////////////////////////////////////
// Vec8f lane load helper used in halo code
///////////////////////////////////////////////////////////////////////////////
template<int pos>
inline Vec8f vload_to(Vec8f& x, float const* p) {
  float tmp[8];
  x.store(tmp);
  tmp[pos] = *p;
  x.load(tmp);
  return x;
}

///////////////////////////////////////////////////////////////////////////////
// Morton helpers
///////////////////////////////////////////////////////////////////////////////
inline Vec4uq _mm256_cvtepi32_epi64(const Vec4i& x) { return Vec4uq(simde_mm256_cvtepi32_epi64(x.v)); }

///////////////////////////////////////////////////////////////////////////////
// Printing (debug only)
///////////////////////////////////////////////////////////////////////////////
inline char* vprint(char* chp, const Vec4f v) {
  float tmp[4]; v.store(tmp);
  sprintf(chp, "%g,%g,%g,%g", tmp[0], tmp[1], tmp[2], tmp[3]);
  return chp;
}
inline char* vprint(char* chp, const Vec8f v) {
  float tmp[8]; v.store(tmp);
  sprintf(chp, "%g,%g,%g,%g,%g,%g,%g,%g", tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]);
  return chp;
}
inline char* vprint(char* chp, const Vec4d v) {
  double tmp[4]; v.store(tmp);
  sprintf(chp, "%g,%g,%g,%g", tmp[0], tmp[1], tmp[2], tmp[3]);
  return chp;
}

#include "vectorclass_util2.h"

#endif  // VECTORCLASS_UTIL_H
