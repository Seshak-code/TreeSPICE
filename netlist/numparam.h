#pragma once
// Parameter front end (ngspice "numparam"): .param expressions, .func definitions,
// parameterized subcircuits and .if/.elseif/.else/.endif, resolved before the netlist
// parser sees the lines. Every {expr}, 'expr' and name=<param> becomes a number; each
// distinct parameter set of a subcircuit becomes one specialized .subckt (equal
// sets share it, so the hierarchy stays reusable); .model cards inside a
// subcircuit are hoisted per specialization.
//
// One expression grammar serves .param and the behavioural sources (B, E/G VALUE, TABLE,
// POLY): it parses to an Expr tree, evaluated on Dual numbers, so a B source gets its
// exact Jacobian from the same code that folds a constant.
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../autodiff/dual.h"

namespace acutesim { namespace numparam {

struct Expr {
    enum Op { Const, Time, Ref, Neg, Not, Add, Sub, Mul, Div, Mod, Pow, Lt, Gt, Le, Ge, Eq, Ne, And, Or, Cond, Fn };
    enum F { SQRT, EXP, LN, LOG10, ABS, FLOOR, CEIL, INT, NINT, SQR, SGN, SIN, COS, TAN, ASIN, ACOS, ATAN, SINH,
             COSH, TANH, ASINH, ACOSH, ATANH, U, URAMP, NOMINAL, POW, PWR, PWRS, ATAN2, HYPOT, MIN, MAX, LIMIT, TABLE };
    struct Node { Op op; double c = 0; F fn = SQRT; std::vector<int> a; };  // Ref: c indexes refs
    struct Sym { char kind; std::string a, b; };  // 'V': V(a) or V(a,b) (nodes); 'I': I(a) (a source), upper case
    std::vector<Node> n;
    std::vector<Sym> refs;
    int root = -1;
    bool runtime() const {  // depends on V(), I() or time
        for (const auto &x : n) if (x.op == Ref || x.op == Time) return true;
        return false;
    }
};

using Lookup = std::function<bool(const std::string &, double &)>;

// Evaluate an expression; lookup(NAME, value) resolves an upper-case identifier.
double eval(const std::string &expr, const Lookup &lookup);

// A behavioural expression (V(), I(), time, signed pwr): the tree, and its canonical
// whitespace-free text (parse(text(e)) is e).
Expr parse(const std::string &expr, const Lookup &lookup);
std::string text(const Expr &e);

// Expand the (line number, text) lines of a netlist whose .include/.lib are already
// spliced. Line 1 (the title) passes through. errLine: the line an error refers to.
std::vector<std::pair<int, std::string>> expand(const std::vector<std::pair<int, std::string>> &lines,
                                                int &errLine, double tempC = 27.0,  // tempC: TEMPER
                                                const std::map<std::string, double> &overrides = {});  // upper-case global .param values that win over the netlist's

// Evaluate node i. Env: time(), ref(k) (the Dual of e.refs[k]), strict (constants: /0 is an error).
template <class Env>
struct Evaluator {
    using D = Dual<double>;
    const Expr &e;
    Env &env;

    D operator()(int i) const {
        const auto &n = e.n[i];
        auto a = [&](size_t k) { return (*this)(n.a[k]); };
        auto b = [](bool x) { return D(x ? 1.0 : 0.0); };
        switch (n.op) {
            case Expr::Const: return D(n.c);
            case Expr::Time: return D(env.time());
            case Expr::Ref: return env.ref((int)n.c);
            case Expr::Neg: return -a(0);
            case Expr::Not: return b(a(0).val == 0.0);
            case Expr::Add: return a(0) + a(1);
            case Expr::Sub: return a(0) - a(1);
            case Expr::Mul: return a(0) * a(1);
            case Expr::Div: { const D d = a(1); if (Env::strict && d.val == 0.0) throw std::runtime_error("division by zero"); return a(0) / d; }
            case Expr::Mod: {
                const D x = a(0), y = a(1);
                if (Env::strict && y.val == 0.0) throw std::runtime_error("division by zero");
                return D(std::fmod(x.val, y.val), x.grad);
            }
            case Expr::Pow: return pow(a(0), a(1));
            case Expr::Lt: return b(a(0).val < a(1).val);
            case Expr::Gt: return b(a(0).val > a(1).val);
            case Expr::Le: return b(a(0).val <= a(1).val);
            case Expr::Ge: return b(a(0).val >= a(1).val);
            case Expr::Eq: return b(a(0).val == a(1).val);
            case Expr::Ne: return b(a(0).val != a(1).val);
            case Expr::And: return b(a(0).val != 0.0 && a(1).val != 0.0);
            case Expr::Or: return b(a(0).val != 0.0 || a(1).val != 0.0);
            case Expr::Cond: return a(0).val != 0.0 ? a(1) : a(2);  // only the taken branch is evaluated
            case Expr::Fn: return call(n);
        }
        return D();
    }

private:
    // x^y with both derivatives; the log only when y varies (a constant exponent of a negative x is fine)
    static D pow(const D &x, const D &y) {
        const double v = std::pow(x.val, y.val);
        double g = 0.0;
        if (x.grad != 0.0) g += y.val * std::pow(x.val, y.val - 1.0) * x.grad;
        if (y.grad != 0.0) g += v * std::log(x.val) * y.grad;
        return D(v, g);
    }
    // ponytail: a slope 1/sqrt(z) at the edge of a domain (sqrt(0), acosh(1)) is capped at 1e12, so a
    // cold start there stays finite; a true singularity then shows as non-convergence
    static double rt(double z) { return std::max(std::sqrt(z), 1e-12); }
    static D flat(double v) { return D(v); }  // piecewise-constant: derivative 0

    D call(const Expr::Node &n) const {
        const D x = (*this)(n.a[0]);
        const double v = x.val, g = x.grad;
        auto arg = [&](size_t k) { return (*this)(n.a[k]); };
        switch (n.fn) {
            case Expr::SQRT: return D(std::sqrt(v), g / (2 * rt(v)));
            case Expr::EXP: { const double s = std::exp(v); return D(s, s * g); }
            case Expr::LN: return D(std::log(v), g / v);
            case Expr::LOG10: return D(std::log10(v), g / (v * 2.302585092994046));
            case Expr::ABS: return D(std::abs(v), ((v > 0) - (v < 0)) * g);
            case Expr::FLOOR: return flat(std::floor(v));
            case Expr::CEIL: return flat(std::ceil(v));
            case Expr::INT: return flat(std::trunc(v));
            case Expr::NINT: return flat(std::nearbyint(v));  // half to even: nint(2.5) = 2
            case Expr::SQR: return x * x;
            case Expr::SGN: return flat((v > 0) - (v < 0));
            case Expr::SIN: return D(std::sin(v), std::cos(v) * g);
            case Expr::COS: return D(std::cos(v), -std::sin(v) * g);
            case Expr::TAN: { const double t = std::tan(v); return D(t, (1 + t * t) * g); }
            case Expr::ASIN: return D(std::asin(v), g / rt(1 - v * v));
            case Expr::ACOS: return D(std::acos(v), -g / rt(1 - v * v));
            case Expr::ATAN: return D(std::atan(v), g / (1 + v * v));
            case Expr::SINH: return D(std::sinh(v), std::cosh(v) * g);
            case Expr::COSH: return D(std::cosh(v), std::sinh(v) * g);
            case Expr::TANH: { const double t = std::tanh(v); return D(t, (1 - t * t) * g); }
            case Expr::ASINH: return D(std::asinh(v), g / std::sqrt(v * v + 1));
            case Expr::ACOSH: return D(std::acosh(v), g / rt(v * v - 1));
            case Expr::ATANH: return D(std::atanh(v), g / (1 - v * v));
            case Expr::U: return flat(v < 0 ? 0.0 : v > 0 ? 1.0 : 0.5);  // ngspice: u(0) = 0.5
            case Expr::URAMP: return v > 0 ? x : flat(0.0);
            case Expr::NOMINAL: return x;  // statistical functions: the nominal value (no Monte Carlo)
            case Expr::POW: return pow(x, arg(1));
            case Expr::PWR: return pow(D(std::abs(v), (v < 0 ? -g : g)), arg(1));  // numparam: |x|^y
            case Expr::PWRS: {  // B sources: sign(x) |x|^y
                const D p = pow(D(std::abs(v), (v < 0 ? -g : g)), arg(1));
                return v < 0 ? -p : p;
            }
            case Expr::ATAN2: {
                const D y = arg(1);  // atan2(x, y)
                return D(std::atan2(v, y.val), (y.val * g - v * y.grad) / (v * v + y.val * y.val));
            }
            case Expr::HYPOT: {
                const D y = arg(1);
                const double h = std::hypot(v, y.val);
                return D(h, (v * g + y.val * y.grad) / h);
            }
            case Expr::MIN: { const D y = arg(1); return v <= y.val ? x : y; }
            case Expr::MAX: { const D y = arg(1); return v >= y.val ? x : y; }
            case Expr::LIMIT: { const D lo = arg(1), hi = arg(2); return v < lo.val ? lo : v > hi.val ? hi : x; }
            case Expr::TABLE: {  // table(x, x0,y0, x1,y1, ...): piecewise linear, clamped to the end values
                const size_t pts = (n.a.size() - 1) / 2;
                auto X = [&](size_t k) { return arg(1 + 2 * k).val; };
                auto Y = [&](size_t k) { return arg(2 + 2 * k).val; };
                if (v <= X(0)) return flat(Y(0));
                if (v >= X(pts - 1)) return flat(Y(pts - 1));
                size_t k = 1;
                while (v > X(k)) ++k;
                const double s = (Y(k) - Y(k - 1)) / (X(k) - X(k - 1));
                return D(Y(k - 1) + s * (v - X(k - 1)), s * g);
            }
        }
        return D();
    }
};

}}  // namespace acutesim::numparam
