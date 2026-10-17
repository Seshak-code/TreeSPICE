#pragma once
#include <cmath>
#include <algorithm>
#include <string>
#include <type_traits>

/**
 * dual.h
 * Lightweight, Header-Only Automatic Differentiation (Forward Mode).
 * 
 * NASA Principle: Static Analysis & Type Safety.
 * This struct allows us to compute f(x) and f'(x) simultaneously.
 * 
 * Usage:
 * Dual<double> x = { 2.0, 1.0 }; // val=2, grad=1 (seeded)
 * Dual<double> y = x * x + 1.0;
 * // y.val  == 5.0
 * // y.grad == 4.0
 */

template <typename T>
struct Dual {
    T val;  // Primal Value
    T grad; // Derivative (Jacobian entry)

    // Constructors
    constexpr Dual() : val(0), grad(0) {}
    constexpr Dual(T v) : val(v), grad(0) {} // Promote scalar to const Dual
    constexpr Dual(T v, T g) : val(v), grad(g) {}

    // Assignment Overloads
    constexpr Dual& operator+=(const Dual& rhs) {
        val += rhs.val;
        grad += rhs.grad;
        return *this;
    }
    
    constexpr Dual& operator-=(const Dual& rhs) {
        val -= rhs.val;
        grad -= rhs.grad;
        return *this;
    }

    constexpr Dual& operator*=(const Dual& rhs) {
        // Product Rule: (u*v)' = u'v + uv'
        grad = grad * rhs.val + val * rhs.grad;
        val *= rhs.val;
        return *this;
    }

    constexpr Dual& operator/=(const Dual& rhs) {
        // Quotient Rule: (u/v)' = (u'v - uv') / v^2
        grad = (grad * rhs.val - val * rhs.grad) / (rhs.val * rhs.val);
        val /= rhs.val;
        return *this;
    }
};

// Binary Operators (Dual op Dual)
template <typename T>
constexpr Dual<T> operator+(Dual<T> lhs, const Dual<T>& rhs) { lhs += rhs; return lhs; }

template <typename T>
constexpr Dual<T> operator-(Dual<T> lhs, const Dual<T>& rhs) { lhs -= rhs; return lhs; }

template <typename T>
constexpr Dual<T> operator*(Dual<T> lhs, const Dual<T>& rhs) { lhs *= rhs; return lhs; }

template <typename T>
constexpr Dual<T> operator/(Dual<T> lhs, const Dual<T>& rhs) { lhs /= rhs; return lhs; }

// Binary Operators (Scalar op Dual)
template <typename T>
constexpr Dual<T> operator+(T lhs, const Dual<T>& rhs) { return Dual<T>(lhs) + rhs; }

template <typename T>
constexpr Dual<T> operator-(T lhs, const Dual<T>& rhs) { return Dual<T>(lhs) - rhs; }

template <typename T>
constexpr Dual<T> operator*(T lhs, const Dual<T>& rhs) { return Dual<T>(lhs) * rhs; }

template <typename T>
constexpr Dual<T> operator/(T lhs, const Dual<T>& rhs) { return Dual<T>(lhs) / rhs; }

// Binary Operators (Dual op Scalar)
template <typename T>
constexpr Dual<T> operator+(Dual<T> lhs, T rhs) { return lhs + Dual<T>(rhs); }

template <typename T>
constexpr Dual<T> operator-(Dual<T> lhs, T rhs) { return lhs - Dual<T>(rhs); }

template <typename T>
constexpr Dual<T> operator*(Dual<T> lhs, T rhs) { return lhs * Dual<T>(rhs); }

template <typename T>
constexpr Dual<T> operator/(Dual<T> lhs, T rhs) { return lhs / Dual<T>(rhs); }

// Comparison Operators (Dual vs Dual)
// NASA Principle: Control flow follows the primal value.
template <typename T>
constexpr bool operator<(const Dual<T>& lhs, const Dual<T>& rhs) { return lhs.val < rhs.val; }
template <typename T>
constexpr bool operator>(const Dual<T>& lhs, const Dual<T>& rhs) { return lhs.val > rhs.val; }
template <typename T>
constexpr bool operator<=(const Dual<T>& lhs, const Dual<T>& rhs) { return lhs.val <= rhs.val; }
template <typename T>
constexpr bool operator>=(const Dual<T>& lhs, const Dual<T>& rhs) { return lhs.val >= rhs.val; }
template <typename T>
constexpr bool operator==(const Dual<T>& lhs, const Dual<T>& rhs) { return lhs.val == rhs.val; }
template <typename T>
constexpr bool operator!=(const Dual<T>& lhs, const Dual<T>& rhs) { return lhs.val != rhs.val; }

// Comparison Operators (Dual vs Scalar)
template <typename T>
constexpr bool operator<(const Dual<T>& lhs, T rhs) { return lhs.val < rhs; }
template <typename T>
constexpr bool operator>(const Dual<T>& lhs, T rhs) { return lhs.val > rhs; }
template <typename T>
constexpr bool operator<=(const Dual<T>& lhs, T rhs) { return lhs.val <= rhs; }
template <typename T>
constexpr bool operator>=(const Dual<T>& lhs, T rhs) { return lhs.val >= rhs; }
template <typename T>
constexpr bool operator==(const Dual<T>& lhs, T rhs) { return lhs.val == rhs; }
template <typename T>
constexpr bool operator!=(const Dual<T>& lhs, T rhs) { return lhs.val != rhs; }

// Comparison Operators (Scalar vs Dual)
template <typename T>
constexpr bool operator<(T lhs, const Dual<T>& rhs) { return lhs < rhs.val; }
template <typename T>
constexpr bool operator>(T lhs, const Dual<T>& rhs) { return lhs > rhs.val; }
template <typename T>
constexpr bool operator<=(T lhs, const Dual<T>& rhs) { return lhs <= rhs.val; }
template <typename T>
constexpr bool operator>=(T lhs, const Dual<T>& rhs) { return lhs >= rhs.val; }
template <typename T>
constexpr bool operator==(T lhs, const Dual<T>& rhs) { return lhs == rhs.val; }
template <typename T>
constexpr bool operator!=(T lhs, const Dual<T>& rhs) { return lhs != rhs.val; }

// Unary Operators
template <typename T>
constexpr Dual<T> operator-(Dual<T> x) { return { -x.val, -x.grad }; }

// Math Functions
// Important: Place in std namespace or allow ADL? Usually ADL is better for T.
// But since Dual is our type, we define overloads in the same namespace or global.

    template <typename T>
    inline Dual<T> exp(const Dual<T>& x) {
        // d/dx exp(u) = exp(u) * u'
        using std::exp;
        T ex = exp(x.val);
        return { ex, ex * x.grad };
    }

    template <typename T>
    inline Dual<T> log(const Dual<T>& x) {
        // d/dx log(u) = u'/u
        using std::log;
        return { log(x.val), x.grad / x.val };
    }

    template <typename T>
    inline Dual<T> sin(const Dual<T>& x) {
        // d/dx sin(u) = cos(u) * u'
        return { std::sin(x.val), std::cos(x.val) * x.grad };
    }

    template <typename T>
    inline Dual<T> cos(const Dual<T>& x) {
        // d/dx cos(u) = -sin(u) * u'
        return { std::cos(x.val), -std::sin(x.val) * x.grad };
    }

    template <typename T>
    inline Dual<T> atan(const Dual<T>& x) {
        return { std::atan(x.val), x.grad / (1 + x.val * x.val) };
    }

    template <typename T>
    inline Dual<T> pow(const Dual<T>& x, T e) {
        using std::pow;
        return { pow(x.val, e), e * pow(x.val, e - 1) * x.grad };
    }

    template <typename T>
    inline Dual<T> sqrt(const Dual<T>& x) {
        // d/dx sqrt(u) = u' / (2*sqrt(u))
        T s = std::sqrt(x.val);
        return { s, x.grad / (2 * s) };
    }
    
    template <typename T>
    inline Dual<T> abs(const Dual<T>& x) {
        // derivative of abs is sgn(x) * x'
        T sgn = (x.val > 0) ? 1 : ((x.val < 0) ? -1 : 0);
        return { std::abs(x.val), sgn * x.grad };
    }
    
    // min/max for clamped functions
    template <typename T>
    inline Dual<T> min(const Dual<T>& a, const Dual<T>& b) {
        return (a.val < b.val) ? a : b; 
    }
    
    template <typename T>
    inline Dual<T> max(const Dual<T>& a, const Dual<T>& b) {
        return (a.val > b.val) ? a : b;
    }
    
    // Scalar min/max
    template <typename T>
    inline Dual<T> min(const Dual<T>& a, T b) {
        return (a.val < b) ? a : Dual<T>(b);
    }
    
    // Overloads for min/max to support mixed scalar/dual
    // Note: The derivative of min(x, C) is 1 if x < C, else 0. 
    // This is correctly handled by returning the full object 'a' or 'b' (which is constant).

