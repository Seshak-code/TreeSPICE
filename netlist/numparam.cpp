#include "numparam.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace acutesim { namespace numparam {

thread_local double temperC = 27.0;  // TEMPER: the circuit temperature of this thread's parse in progress

namespace {

std::string upper(std::string s) {
    for (auto &c : s) c = (char)std::toupper((unsigned char)c);
    return s;
}
bool identStart(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
bool identChar(char c) { return std::isalnum((unsigned char)c) || c == '_' || c == '.'; }
std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
std::string num(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.17g", v);
    return b;
}

// ---------------------------------------------------------------- expressions
struct Func {  // .func name(params) body: inlined at each call, so arguments may be V(), I() or time
    std::vector<std::string> params;
    std::string body;
};
using FuncLookup = std::function<const Func *(const std::string &)>;

struct FnInfo { const char *name; Expr::F f; size_t lo, hi; };
const FnInfo kFns[] = {
    {"SQRT", Expr::SQRT, 1, 1}, {"EXP", Expr::EXP, 1, 1}, {"LN", Expr::LN, 1, 1}, {"LOG", Expr::LN, 1, 1},
    {"LOG10", Expr::LOG10, 1, 1}, {"ABS", Expr::ABS, 1, 1}, {"FLOOR", Expr::FLOOR, 1, 1}, {"CEIL", Expr::CEIL, 1, 1},
    {"INT", Expr::INT, 1, 1}, {"NINT", Expr::NINT, 1, 1}, {"SQR", Expr::SQR, 1, 1}, {"SGN", Expr::SGN, 1, 1},
    {"SIGN", Expr::SGN, 1, 1}, {"SIN", Expr::SIN, 1, 1}, {"COS", Expr::COS, 1, 1}, {"TAN", Expr::TAN, 1, 1},
    {"ASIN", Expr::ASIN, 1, 1}, {"ACOS", Expr::ACOS, 1, 1}, {"ATAN", Expr::ATAN, 1, 1}, {"SINH", Expr::SINH, 1, 1},
    {"COSH", Expr::COSH, 1, 1}, {"TANH", Expr::TANH, 1, 1}, {"ASINH", Expr::ASINH, 1, 1},
    {"ACOSH", Expr::ACOSH, 1, 1}, {"ATANH", Expr::ATANH, 1, 1}, {"U", Expr::U, 1, 1}, {"URAMP", Expr::URAMP, 1, 1},
    {"POW", Expr::POW, 2, 2}, {"PWR", Expr::PWR, 2, 2}, {"ATAN2", Expr::ATAN2, 2, 2}, {"HYPOT", Expr::HYPOT, 2, 2},
    {"MIN", Expr::MIN, 2, 2}, {"MAX", Expr::MAX, 2, 2},
    // Statistical functions return their nominal value: no Monte Carlo yet.
    // ponytail: agauss/gauss/aunif/unif/limit(nom, var) are deterministic (nominal); a corner
    // that switches them on (e.g. sky130 mismatch with mc_mm_switch=1) is not random.
    {"AGAUSS", Expr::NOMINAL, 2, 3}, {"GAUSS", Expr::NOMINAL, 2, 3}, {"AUNIF", Expr::NOMINAL, 2, 3},
    {"UNIF", Expr::NOMINAL, 2, 3},
    {"LIMIT", Expr::LIMIT, 3, 3},  // clamp(x, lo, hi); the 2-argument limit(nom, var) is NOMINAL
    {"TABLE", Expr::TABLE, 5, 1000},  // table(x, x0,y0, x1,y1, ...): what `E.. TABLE {x} = (x0,y0) ...` becomes
};
const char *fnName(Expr::F f) {
    if (f == Expr::PWRS) f = Expr::PWR;
    for (const auto &i : kFns) if (i.f == f) return i.name;
    return "";
}

struct Ctx {  // shared by a parse and the .func bodies it inlines
    const Lookup &look;
    const FuncLookup &funcs;
    bool beh;  // behavioural: V(), I(), time, and pwr keeps the sign of its base (ngspice's B sources)
    Expr &e;
    int depth = 0;
};

class Parser {
public:
    Parser(const std::string &s, Ctx &c, std::map<std::string, int> bind = {}) : s_(s), c_(c), bind_(std::move(bind)) {}
    int run() {
        const int v = ternary();
        ws();
        if (p_ != s_.size()) fail("unexpected '" + s_.substr(p_) + "'");
        return v;
    }

private:
    const std::string &s_;
    Ctx &c_;
    std::map<std::string, int> bind_;  // .func parameter -> the node of the argument
    size_t p_ = 0;

    [[noreturn]] void fail(const std::string &why) { throw std::runtime_error(why); }
    int mk(Expr::Op op, std::vector<int> a = {}, double c = 0, Expr::F fn = Expr::SQRT) {
        c_.e.n.push_back({op, c, fn, std::move(a)});
        return (int)c_.e.n.size() - 1;
    }
    void ws() { while (p_ < s_.size() && std::isspace((unsigned char)s_[p_])) ++p_; }
    bool eat(const char *op) {
        ws();
        const size_t n = std::char_traits<char>::length(op);
        if (s_.compare(p_, n, op) != 0) return false;
        p_ += n;
        return true;
    }
    int ternary() {
        const int c = orExpr();
        if (!eat("?")) return c;
        const int a = ternary();
        if (!eat(":")) fail("'?' without ':'");
        const int b = ternary();
        return mk(Expr::Cond, {c, a, b});
    }
    int orExpr() {
        int v = andExpr();
        while (eat("||")) v = mk(Expr::Or, {v, andExpr()});
        return v;
    }
    int andExpr() {
        int v = cmp();
        while (eat("&&")) v = mk(Expr::And, {v, cmp()});
        return v;
    }
    int cmp() {
        int v = add();
        for (;;) {
            if (eat("==")) v = mk(Expr::Eq, {v, add()});
            else if (eat("!=") || eat("<>")) v = mk(Expr::Ne, {v, add()});
            else if (eat("<=")) v = mk(Expr::Le, {v, add()});
            else if (eat(">=")) v = mk(Expr::Ge, {v, add()});
            else if (eat("<")) v = mk(Expr::Lt, {v, add()});
            else if (eat(">")) v = mk(Expr::Gt, {v, add()});
            else return v;
        }
    }
    int add() {
        int v = mul();
        for (;;) {
            if (eat("+")) v = mk(Expr::Add, {v, mul()});
            else if (eat("-")) v = mk(Expr::Sub, {v, mul()});
            else return v;
        }
    }
    int mul() {
        int v = unary();
        for (;;) {
            ws();
            if (s_.compare(p_, 2, "**") == 0) return v;  // power, handled in unary's pow()
            if (eat("*")) v = mk(Expr::Mul, {v, unary()});
            else if (eat("/")) v = mk(Expr::Div, {v, unary()});
            else if (eat("%")) v = mk(Expr::Mod, {v, unary()});
            else return v;
        }
    }
    int unary() {
        if (eat("-")) return mk(Expr::Neg, {unary()});
        if (eat("+")) return unary();
        if (eat("!")) return mk(Expr::Not, {unary()});
        int b = primary();  // ** and ^ are left-associative in ngspice (2**3**2 = 64),
        while (eat("**") || eat("^")) b = mk(Expr::Pow, {b, signedPrimary()});  // binding tighter than unary -
        return b;
    }
    int signedPrimary() {  // an exponent: 2**-1
        if (eat("-")) return mk(Expr::Neg, {signedPrimary()});
        if (eat("+")) return signedPrimary();
        return primary();
    }
    int number() {
        const char *start = s_.c_str() + p_;
        char *end = nullptr;
        double v = std::strtod(start, &end);
        p_ += (size_t)(end - start);
        std::string suf;  // SI scale factor, then unit letters (ignored)
        while (p_ < s_.size() && std::isalpha((unsigned char)s_[p_])) suf += (char)std::toupper((unsigned char)s_[p_++]);
        if (suf.rfind("MEG", 0) == 0) v *= 1e6;  // MIL is milli here, as ngspice's numparam
        else if (!suf.empty())
            switch (suf[0]) {
                case 'T': v *= 1e12; break;
                case 'G': v *= 1e9; break;
                case 'K': v *= 1e3; break;
                case 'M': v *= 1e-3; break;
                case 'U': v *= 1e-6; break;
                case 'N': v *= 1e-9; break;
                case 'P': v *= 1e-12; break;
                case 'F': v *= 1e-15; break;
                case 'A': v *= 1e-18; break;
                default: break;  // unit letters only (e.g. 10V)
            }
        return mk(Expr::Const, {}, v);
    }
    int call(const std::string &f, const std::vector<int> &a) {
        if (const Func *u = c_.funcs ? c_.funcs(f) : nullptr) {  // .func: inline the body over the arguments
            if (a.size() != u->params.size()) fail(f + "() takes " + std::to_string(u->params.size()) + " argument(s)");
            if (++c_.depth > 32) fail(".func " + f + " calls itself");
            std::map<std::string, int> bind;
            for (size_t i = 0; i < a.size(); ++i) bind[u->params[i]] = a[i];
            const int r = Parser(u->body, c_, std::move(bind)).run();
            --c_.depth;
            return r;
        }
        for (const auto &i : kFns) {
            if (f != i.name) continue;
            Expr::F fn = i.f;
            size_t lo = i.lo;
            if (fn == Expr::LIMIT && a.size() == 2) { fn = Expr::NOMINAL; lo = 2; }  // limit(nominal, variation)
            if (fn == Expr::PWR && c_.beh) fn = Expr::PWRS;
            if (a.size() < lo || a.size() > i.hi || (fn == Expr::TABLE && a.size() % 2 == 0))
                fail(f + "(): wrong argument count");
            return mk(Expr::Fn, a, 0, fn);
        }
        fail("function " + f + "() is not supported yet");
    }
    int primary() {
        ws();
        if (p_ >= s_.size()) fail("unexpected end");
        if (eat("(")) {
            const int v = ternary();
            if (!eat(")")) fail("missing ')'");
            return v;
        }
        if (eat("{")) {  // {expr} nested inside a quoted expression
            const int v = ternary();
            if (!eat("}")) fail("missing '}'");
            return v;
        }
        const char c = s_[p_];
        if (std::isdigit((unsigned char)c) || c == '.') return number();
        if (!identStart(c)) fail(std::string("unexpected '") + c + "'");
        size_t b = p_;
        while (p_ < s_.size() && identChar(s_[p_])) ++p_;
        const std::string name = upper(s_.substr(b, p_ - b));
        if ((name == "V" || name == "I") && eat("(")) return ref(name[0]);
        if (eat("(")) {
            std::vector<int> args;
            if (!eat(")")) {
                do args.push_back(ternary()); while (eat(","));
                if (!eat(")")) fail("missing ')' after " + name + " arguments");
            }
            return call(name, args);
        }
        if (auto it = bind_.find(name); it != bind_.end()) return it->second;
        if (name == "PI") return mk(Expr::Const, {}, M_PI);
        if (name == "TEMPER") return mk(Expr::Const, {}, temperC);  // circuit temperature in C
        if (name == "TIME" && c_.beh) return mk(Expr::Time);
        double v;
        if (c_.look(name, v)) return mk(Expr::Const, {}, v);
        fail("unknown parameter " + name);
    }
    // V(a), V(a,b) and I(src) after the "(": node and source names, not expressions
    int ref(char kind) {
        if (!c_.beh) fail(std::string(1, kind) + "() is only available in B, E and G expressions");
        const size_t close = s_.find(')', p_);
        if (close == std::string::npos) fail("missing ')' after " + std::string(1, kind) + "(");
        std::vector<std::string> names;
        std::istringstream in(s_.substr(p_, close - p_));
        for (std::string x; std::getline(in, x, ',');) names.push_back(upper(trim(x)));
        p_ = close + 1;
        if (names.empty() || names.size() > (kind == 'V' ? 2u : 1u) || names[0].empty() || (names.size() > 1 && names[1].empty()))
            fail(std::string(1, kind) + "() takes " + (kind == 'V' ? "one or two nodes" : "one source name"));
        Expr::Sym r{kind, names[0], names.size() > 1 ? names[1] : ""};
        auto &refs = c_.e.refs;
        size_t k = 0;
        while (k < refs.size() && !(refs[k].kind == r.kind && refs[k].a == r.a && refs[k].b == r.b)) ++k;
        if (k == refs.size()) refs.push_back(r);
        return mk(Expr::Ref, {}, (double)k);
    }
};

struct ConstEnv {  // constants only
    static constexpr bool strict = true;  // ngspice errors on /0
    double time() const { throw std::runtime_error("unknown parameter TIME"); }
    Dual<double> ref(int) const { throw std::runtime_error("V() and I() are only available in B, E and G expressions"); }
};

// text -> tree. A failure names the expression.
Expr build(const std::string &text, const Lookup &look, const FuncLookup &funcs, bool beh) {
    Expr e;
    Ctx c{look, funcs, beh, e};
    try { e.root = Parser(text, c).run(); }
    catch (const std::runtime_error &x) { throw std::runtime_error("expression '" + text + "': " + x.what()); }
    return e;
}

// A constant's value; ngspice errors on a result that is not a finite number (ln(0), sqrt(-1), 1e400).
double value(const Expr &e, const std::string &text) {
    ConstEnv env;
    try {
        const double v = Evaluator<ConstEnv>{e, env}(e.root).val;
        if (!std::isfinite(v)) throw std::runtime_error("the result is not a finite number");
        return v;
    } catch (const std::runtime_error &x) { throw std::runtime_error("expression '" + text + "': " + x.what()); }
}

void print(const Expr &e, int i, std::string &out) {
    const auto &n = e.n[i];
    static const char *bin[] = {"+", "-", "*", "/", "%", "^", "<", ">", "<=", ">=", "==", "!=", "&&", "||"};
    auto sub = [&](size_t k) { print(e, n.a[k], out); };
    switch (n.op) {
        case Expr::Const: out += n.c < 0 ? "(" + num(n.c) + ")" : num(n.c); break;
        case Expr::Time: out += "time"; break;
        case Expr::Ref: {
            const auto &r = e.refs[(size_t)n.c];
            out += std::string(1, r.kind) + "(" + r.a + (r.b.empty() ? "" : "," + r.b) + ")";
            break;
        }
        case Expr::Neg: case Expr::Not: out += n.op == Expr::Neg ? "(-" : "(!"; sub(0); out += ")"; break;
        case Expr::Cond: out += "("; sub(0); out += "?"; sub(1); out += ":"; sub(2); out += ")"; break;
        case Expr::Fn:
            out += std::string(fnName(n.fn)) + "(";
            for (size_t k = 0; k < n.a.size(); ++k) { if (k) out += ","; sub(k); }
            out += ")";
            break;
        default: out += "("; sub(0); out += bin[n.op - Expr::Add]; sub(1); out += ")";
    }
}

// ---------------------------------------------------------------- scopes
// Parameters by upper-case name: raw expressions, evaluated lazily (forward
// references work; a cycle is an error), then cached. Parent: the enclosing scope.
struct Scope {
    std::map<std::string, std::string> raw;
    std::map<std::string, int> line;  // where each raw parameter was defined
    std::map<std::string, double> val;
    std::map<std::string, Func> funcs;  // .func definitions (upper-case names)
    std::set<std::string> busy;
    Scope *parent = nullptr;
    const void *owner = nullptr;                      // the Subckt whose body this scope holds
    std::string tag;                                  // that instance's emitted subcircuit name
    std::map<std::string, std::string> models;        // its local .model names -> hoisted names

    bool lookup(const std::string &n, double &v) {
        if (auto it = val.find(n); it != val.end()) { v = it->second; return true; }
        if (auto it = raw.find(n); it != raw.end()) {
            if (!busy.insert(n).second) throw std::runtime_error("parameter " + n + " depends on itself");
            v = evaluate(it->second);
            busy.erase(n);
            val[n] = v;
            return true;
        }
        return parent && parent->lookup(n, v);
    }
    bool has(const std::string &n) const {
        return val.count(n) || raw.count(n) || (parent && parent->has(n));
    }
    const Func *func(const std::string &n) const {
        if (auto it = funcs.find(n); it != funcs.end()) return &it->second;
        return parent ? parent->func(n) : nullptr;
    }
    static std::string unwrap(std::string e) {
        e = trim(e);
        if (e.size() >= 2 && ((e.front() == '{' && e.back() == '}') || (e.front() == '\'' && e.back() == '\'')))
            e = e.substr(1, e.size() - 2);
        return e;
    }
    Lookup lookupFn() { return [this](const std::string &n, double &v) { return lookup(n, v); }; }
    FuncLookup funcFn() const { return [this](const std::string &n) { return func(n); }; }
    double evaluate(const std::string &text) {
        const std::string e = unwrap(text);
        return value(build(e, lookupFn(), funcFn(), false), e);
    }
    // A B/E/G expression stays symbolic (V(), I(), time are runtime); parameters and .func calls
    // fold in, and the canonical text is what the netlist parser reads back. A constant is its value.
    std::string behavioural(const std::string &src) {
        std::string e = trim(src);
        if (e.size() >= 2 && e.front() == '{') {  // {expr}: the brace closing the first one ends the text
            int d = 0;
            size_t j = 0;
            for (; j < e.size(); ++j) { d += (e[j] == '{') - (e[j] == '}'); if (!d) break; }
            if (j == e.size() - 1) e = trim(e.substr(1, e.size() - 2));
        } else if (e.size() >= 2 && e.front() == '\'' && e.back() == '\'' && e.find('\'', 1) == e.size() - 1)
            e = trim(e.substr(1, e.size() - 2));
        const Expr x = build(e, lookupFn(), funcFn(), true);
        return x.runtime() ? numparam::text(x) : num(value(x, e));
    }
};

// name = value pairs of text (".param a=1 b='x*2' c = {y}", subckt/X parameters):
// a value runs to the next top-level "name =" (quotes, braces and parens may hold
// spaces). first: offset where the first pair starts (text.size() if none).
std::vector<std::pair<std::string, std::string>> pairs(const std::string &t, size_t *first = nullptr) {
    std::vector<size_t> starts;
    int depth = 0;
    bool q = false;
    for (size_t i = 0; i < t.size(); ++i) {
        const char c = t[i];
        if (c == '\'') q = !q;
        if (q) continue;
        if (c == '{' || c == '(') ++depth;
        if (c == '}' || c == ')') --depth;
        if (depth || !identStart(c) || (i && identChar(t[i - 1]))) continue;
        size_t j = i;
        while (j < t.size() && identChar(t[j])) ++j;
        size_t k = j;
        while (k < t.size() && t[k] == ' ') ++k;
        if (k < t.size() && t[k] == '=' && (k + 1 >= t.size() || t[k + 1] != '=')) starts.push_back(i);
        i = j - 1;
    }
    if (first) *first = starts.empty() ? t.size() : starts[0];
    std::vector<std::pair<std::string, std::string>> out;
    for (size_t s = 0; s < starts.size(); ++s) {
        const size_t eq = t.find('=', starts[s]);
        const size_t end = s + 1 < starts.size() ? starts[s + 1] : t.size();
        std::string v = trim(t.substr(eq + 1, end - eq - 1));
        // "r=3k PARAMS: q=2": a params: keyword between pairs ends up after the previous value
        if (v.size() > 7 && upper(v.substr(v.size() - 7)) == "PARAMS:" && std::isspace((unsigned char)v[v.size() - 8]))
            v = trim(v.substr(0, v.size() - 7));
        out.push_back({upper(trim(t.substr(starts[s], eq - starts[s]))), v});
    }
    return out;
}
// The words before the first "name =" pair, without a "params:" keyword.
std::vector<std::string> headWords(const std::string &t) {
    size_t first;
    pairs(t, &first);
    std::vector<std::string> w;
    std::istringstream in(t.substr(0, first));
    for (std::string x; in >> x;)
        if (upper(x) != "PARAMS:") w.push_back(x);
    return w;
}
std::vector<std::string> words(const std::string &t) {
    std::istringstream in(t);
    std::vector<std::string> w;
    for (std::string x; in >> x;) w.push_back(x);
    return w;
}

// {expr} and 'expr' -> number; name=<param in scope> -> name=number.
std::string substitute(const std::string &line, Scope &sc) {
    std::string out;
    for (size_t i = 0; i < line.size();) {
        const char c = line[i];
        if (c == '{' || c == '\'') {
            const char close = c == '{' ? '}' : '\'';
            int depth = 0;
            size_t j = i;
            for (; j < line.size(); ++j) {
                if (c == '{' && line[j] == '{') ++depth;
                if (line[j] == close && (c == '\'' ? j > i : --depth == 0)) break;
            }
            if (j >= line.size()) throw std::runtime_error(std::string("unbalanced ") + c + " in: " + line);
            out += num(sc.evaluate(line.substr(i, j - i + 1)));
            i = j + 1;
            continue;
        }
        if (c == '=' ) {
            out += c;
            size_t j = i + 1;
            while (j < line.size() && line[j] == ' ') out += line[j++];
            size_t k = j;
            if (k < line.size() && identStart(line[k])) {
                while (k < line.size() && identChar(line[k])) ++k;
                const std::string id = upper(line.substr(j, k - j));
                double v;
                if ((k == line.size() || line[k] == ' ' || line[k] == ')' || line[k] == ',') && sc.has(id) && sc.lookup(id, v)) {
                    out += num(v);
                    i = k;
                    continue;
                }
            }
            i = j;
            continue;
        }
        out += c;
        ++i;
    }
    return out;
}

// B cards, and E/G cards in a VALUE / VOL / CUR / TABLE form (POLY and the linear form are numbers).
bool isBehavioural(const std::vector<std::string> &w) {
    const char t = (char)std::toupper((unsigned char)w[0][0]);
    if (t == 'B') return true;
    if ((t != 'E' && t != 'G') || w.size() < 4) return false;
    const std::string k = upper(w[3]);
    for (const char *kw : {"VALUE", "VOL", "CUR", "TABLE"})
        if (k.rfind(kw, 0) == 0 && (k.size() == std::strlen(kw) || k[std::strlen(kw)] == '=' || k[std::strlen(kw)] == '{' || k[std::strlen(kw)] == '\''))
            return true;
    return false;
}

// "B1 n+ n- V=expr", "E1 n+ n- VALUE={expr}", "E1 n+ n- TABLE {expr} = (x,y) ...": the expression becomes
// canonical text (no spaces) after the "=" or "TABLE"; the table points get the ordinary {} / '' treatment.
std::string behaviouralLine(const std::string &line, Scope &sc) {
    size_t p = 0;  // end of the third word
    for (int i = 0; i < 3; ++i) {
        p = line.find_first_not_of(" \t", p);
        p = p == std::string::npos ? p : line.find_first_of(" \t", p);
        if (p == std::string::npos) throw std::runtime_error("expected name, n+, n-, then the expression");
    }
    const size_t k0 = line.find_first_not_of(" \t", p);
    size_t k1 = k0;
    while (k1 != std::string::npos && k1 < line.size() && identChar(line[k1])) ++k1;
    if (k0 == std::string::npos) throw std::runtime_error("no expression");
    const std::string kw = upper(line.substr(k0, k1 - k0)), head = line.substr(0, p);
    if (kw == "TABLE") {
        int depth = 0;
        bool q = false;
        for (size_t i = k1; i < line.size(); ++i) {  // the top-level "=" (not ==, <=, >=, !=)
            const char c = line[i];
            if (c == '\'') q = !q;
            if (q) continue;
            depth += (c == '(' || c == '{') - (c == ')' || c == '}');
            if (depth == 0 && c == '=' && (i + 1 >= line.size() || line[i + 1] != '=') && !std::strchr("<>!=", line[i - 1]))
                return head + " TABLE " + sc.behavioural(line.substr(k1, i - k1)) + " = " + substitute(line.substr(i + 1), sc);
        }
        throw std::runtime_error("TABLE needs {expr} = (x,y) ...");
    }
    const size_t eq = line.find_first_not_of(" \t", k1);
    if (eq == std::string::npos || line[eq] != '=') throw std::runtime_error(kw + " must be followed by '='");
    return head + " " + kw + "=" + sc.behavioural(line.substr(eq + 1));
}

using Lines = std::vector<std::pair<int, std::string>>;

struct Subckt {
    std::string name;
    std::vector<std::string> ports;
    std::vector<std::pair<std::string, std::string>> defaults;
    Lines body;
    const Subckt *parent = nullptr;     // the .subckt this one is nested in (null: top level)
    std::map<std::string, Subckt> kids; // nested definitions, visible in this body and below
};

class Expander {
public:
    Expander(int &errLine, const std::map<std::string, double> &overrides) : errLine_(errLine), overrides_(overrides) {}

    Lines run(const Lines &in) {
        Lines top;
        for (size_t i = 0; i < in.size(); ++i) {  // collect .subckt bodies; the rest is the top level
            errLine_ = in[i].first;
            const auto w = words(in[i].second);
            if (in[i].first == 1 || w.empty() || upper(w[0]) != ".SUBCKT") { top.push_back(in[i]); continue; }
            readSubckt(in, i, subckts_, nullptr);
        }
        for (const auto &[n, l] : top) { errLine_ = n; collectParams(l, global_, n); }
        for (const auto &[name, v] : overrides_) { global_.raw[name] = num(v); global_.line.emplace(name, 0); }  // .step param
        Lines out = walk(top, global_, "", nullptr, nullptr, "");
        Lines params;  // the evaluated global parameters, for the parser's {NAME} / bare-name values
        for (const auto &[name, e] : global_.raw) {
            errLine_ = global_.line[name];
            double v;
            global_.lookup(name, v);
            params.push_back({0, ".param " + name + "=" + num(v)});
        }
        size_t end = out.size();  // definitions go before .end (the parser stops there)
        for (size_t i = 0; i < out.size(); ++i)
            if (const auto w = words(out[i].second); i && !w.empty() && upper(w[0]) == ".END") { end = i; break; }
        out.insert(out.begin() + (long)std::min<size_t>(1, out.size()), params.begin(), params.end());
        end += params.size();
        out.insert(out.begin() + (long)end, defs_.begin(), defs_.end());
        return out;
    }

private:
    int &errLine_;
    const std::map<std::string, double> &overrides_;
    std::map<std::string, Subckt> subckts_;
    // The .subckt at in[i] up to its .ends, nested definitions into its kids; i ends on the .ends
    void readSubckt(const Lines &in, size_t &i, std::map<std::string, Subckt> &into, const Subckt *parent) {
        const std::string &card = in[i].second;
        const std::string hdr = card.substr(card.find(words(card)[0]) + 7);
        const auto hw = headWords(hdr);
        if (hw.empty()) throw std::runtime_error(".subckt needs a name");
        auto [it, isNew] = into.try_emplace(upper(hw[0]));
        if (!isNew) throw std::runtime_error("subcircuit " + upper(hw[0]) + " defined twice");
        Subckt &s = it->second;
        s.name = upper(hw[0]);
        s.ports = {hw.begin() + 1, hw.end()};
        s.defaults = pairs(hdr);
        s.parent = parent;
        for (++i; i < in.size(); ++i) {
            const auto bw = words(in[i].second);
            const std::string k = bw.empty() ? "" : upper(bw[0]);
            if (k == ".ENDS") return;
            errLine_ = in[i].first;
            if (k == ".SUBCKT") readSubckt(in, i, s.kids, &s);
            else s.body.push_back(in[i]);
        }
        throw std::runtime_error(".subckt " + s.name + " has no .ends");
    }
    // The subcircuit an X card names: a nested definition of the enclosing bodies, innermost first, else a top-level one
    const Subckt *find(const std::string &name, const Subckt *ctx) const {
        for (; ctx; ctx = ctx->parent)
            if (auto it = ctx->kids.find(name); it != ctx->kids.end()) return &it->second;
        const auto it = subckts_.find(name);
        return it == subckts_.end() ? nullptr : &it->second;
    }
    std::map<std::string, std::string> specialized_;  // key -> emitted subckt name
    Scope global_;
    Lines defs_;  // specialized .subckt definitions and their hoisted .model cards

    // .param name=expr ... and .func name(args) {expr}, into scope sc
    void collectParams(const std::string &l, Scope &sc, int n = 0) {
        const auto w = words(l);
        if (w.empty()) return;
        const std::string k = upper(w[0]), rest = l.substr(l.find(w[0]) + w[0].size());
        if (k == ".PARAM" || k == ".PARAMS") {
            for (auto &[name, v] : pairs(rest)) { sc.raw[name] = v; sc.line[name] = n; }
        } else if (k == ".FUNC") {
            const size_t lp = rest.find('('), rp = rest.find(')');
            if (lp == std::string::npos || rp == std::string::npos || rp < lp)
                throw std::runtime_error(".func needs name(args) {expression}");
            Func f;
            std::istringstream in(rest.substr(lp + 1, rp - lp - 1));
            for (std::string a; std::getline(in, a, ',');) if (!trim(a).empty()) f.params.push_back(upper(trim(a)));
            std::string body = trim(rest.substr(rp + 1));
            if (!body.empty() && body[0] == '=') body = body.substr(1);
            f.body = Scope::unwrap(body);
            sc.funcs[upper(trim(rest.substr(0, lp)))] = std::move(f);
        }
    }

    // Lines of one scope: .if resolved, .param consumed, X cards specialized,
    // expressions substituted. localModels: renames for hoisted local .model cards. ctx: the .subckt whose
    // body this is (for nested definitions). mult: the instance's m=, set on every device that has one.
    Lines walk(const Lines &in, Scope &sc, const std::string &prefix, std::map<std::string, std::string> *localModels,
               const Subckt *ctx, const std::string &mult) {
        Lines out;
        std::vector<std::pair<bool, bool>> ifs;  // (this branch active, some branch taken)
        std::vector<bool> elses;                 // per open .if: its .else seen
        auto active = [&] { for (auto &f : ifs) if (!f.first) return false; return true; };
        for (const auto &[n, raw] : in) {
            errLine_ = n;
            if (n == 1 && prefix.empty()) { out.push_back({n, raw}); continue; }
            const auto w = words(raw);
            if (w.empty()) continue;
            const std::string k = upper(w[0]);
            const std::string rest = raw.substr(raw.find(w[0]) + w[0].size());
            if (k == ".IF") {
                const bool on = active() && sc.evaluate(rest) != 0.0;
                ifs.push_back({on, on});
                elses.push_back(false);
                continue;
            }
            if (k == ".ELSEIF" || k == ".ELSE") {
                if (ifs.empty()) throw std::runtime_error(k + " without .if");
                if (elses.back()) throw std::runtime_error(k + " after .else");
                elses.back() = k == ".ELSE";
                const auto f = ifs.back();  // a copy: pop_back ends the element's lifetime
                ifs.pop_back();
                const bool outer = active();
                const bool on = outer && !f.second && (k == ".ELSE" || sc.evaluate(rest) != 0.0);
                ifs.push_back({on, f.second || on});
                continue;
            }
            if (k == ".ENDIF") {
                if (ifs.empty()) throw std::runtime_error(".endif without .if");
                ifs.pop_back();
                elses.pop_back();
                continue;
            }
            if (!active()) continue;
            if (k == ".PARAM" || k == ".PARAMS" || k == ".FUNC") { if (!prefix.empty()) collectParams(raw, sc); continue; }  // top level: collected
            if (k == ".OSDI" || k == ".INCLUDE" || k == ".INC" || k == ".LIB" || k == ".TITLE") { out.push_back({n, raw}); continue; }
            std::string line = raw;
            if (localModels)  // names of this subcircuit's own models -> hoisted names
                for (const auto &[from, to] : *localModels) line = renameWord(line, from, to);
            line = isBehavioural(w) ? behaviouralLine(line, sc) : substitute(line, sc);
            if (k == ".MODEL" && localModels) {
                defs_.push_back({n, line});
                continue;
            }
            if (!mult.empty() && std::strchr("RCLIDQMJGFX", k[0])) {  // the instance's m= replaces any of the device's own (ngspice)
                static const std::regex own(R"((^|\s)[mM]\s*=\s*[^\s]+)");
                line = std::regex_replace(line, own, "$1") + " m=" + mult;
            }
            if (k[0] == 'N' && !mult.empty())
                throw std::runtime_error(w[0] + ": m= on a subcircuit holding an OSDI device is not supported yet");
            if (k[0] == 'X') line = instance(line, sc, n, ctx);
            out.push_back({n, line});
        }
        if (!ifs.empty()) throw std::runtime_error(".if without .endif");
        return out;
    }

    static std::string renameWord(const std::string &line, const std::string &from, const std::string &to) {
        std::string out;
        const std::string up = upper(line);
        for (size_t i = 0; i < line.size();) {
            if (up.compare(i, from.size(), from) == 0 && (i == 0 || !identChar(line[i - 1])) &&
                (i + from.size() == line.size() || !identChar(line[i + from.size()]))) {
                out += to;
                i += from.size();
            } else out += line[i++];
        }
        return out;
    }

    // X<name> <nodes...> <subckt> [params:] [k=v ...] -> X<name> <nodes...> <specialized subckt>
    std::string instance(const std::string &line, Scope &caller, int n, const Subckt *ctx) {
        const auto hw = headWords(line);
        if (hw.size() < 2) throw std::runtime_error(hw.empty() ? "empty X card" : hw[0] + ": no subcircuit name");
        const std::string name = upper(hw.back());
        const Subckt *found = find(name, ctx);
        if (!found) return line;  // the parser reports the unknown subcircuit
        const Subckt &s = *found;
        Scope sc;
        sc.parent = &global_;
        Scope *lex = nullptr;  // a nested definition sees the parameters and models of the instance it sits in
        if (s.parent) {
            for (lex = &caller; lex && lex->owner != s.parent; lex = lex->parent) {}
            if (!lex) throw std::runtime_error(hw[0] + ": " + s.name + " is not visible here");
            sc.parent = lex;
            sc.models = lex->models;
        }
        sc.owner = &s;
        for (const auto &[k, v] : s.defaults) sc.raw[k] = v;
        // The body's own .param values are defaults too: an X card overrides them
        // (ngspice; sky130 declares l, w, ... that way). Overridden values win
        // because a scope reads val before raw.
        Scope body;
        for (const auto &[bn, bl] : s.body) collectParams(bl, body);
        std::string given, mult;
        for (const auto &[k, v] : pairs(line)) {
            if (k == "M" && !sc.raw.count(k) && !body.raw.count(k)) {  // ngspice scales every device inside
                mult = num(caller.evaluate(v));
                continue;
            }
            if (!sc.raw.count(k) && !body.raw.count(k)) throw std::runtime_error(hw[0] + ": " + s.name + " has no parameter " + k);
            sc.raw.erase(k);
            sc.val[k] = caller.evaluate(v);
            if (body.raw.count(k)) given += " " + k + "=" + num(sc.val[k]);
        }
        std::string key = (s.parent ? s.name + "@" + lex->tag : s.name) + given + (mult.empty() ? "" : " m=" + mult);
        for (const auto &[k, v] : s.defaults) { double x; sc.lookup(k, x); key += " " + k + "=" + num(x); }
        // Generated names use '|', which SPICE names never hold (sky130's hold "__"): the parser's
        // messages map them back to the user's names (SPICEParser::parse, fail). Only a key with no
        // parameters keeps the plain name: body-.param subckts (sky130) differ by `given` alone
        auto [spec, isNew] = specialized_.emplace(key, key == s.name && !s.parent ? s.name : s.name + "|" + std::to_string(specialized_.size() + 1));
        sc.tag = spec->second;
        if (isNew) {
            for (const auto &[bn, bl] : s.body) collectParams(bl, sc);  // body .params (any order)
            std::map<std::string, std::string> &models = sc.models;
            for (const auto &[bn, bl] : s.body)
                if (const auto w = words(bl); w.size() > 1 && upper(w[0]) == ".MODEL") {
                    const std::string m = upper(w[1]);
                    models[m] = spec->second + "|" + m;
                    if (const size_t dot = m.rfind('.'); dot != std::string::npos)  // binned family NAME.N:
                        models[m.substr(0, dot)] = spec->second + "|" + m.substr(0, dot);  // devices name NAME
                }
            std::string hdr = ".subckt " + spec->second;
            for (const auto &p : s.ports) hdr += " " + p;
            Lines body = walk(s.body, sc, spec->second, &models, &s, mult);
            errLine_ = n;
            defs_.push_back({s.body.empty() ? n : s.body.front().first, hdr});
            defs_.insert(defs_.end(), body.begin(), body.end());
            defs_.push_back({s.body.empty() ? n : s.body.back().first, ".ends"});
        }
        std::string out;
        for (size_t i = 0; i + 1 < hw.size(); ++i) out += hw[i] + " ";
        return out + spec->second;
    }
};

}  // namespace

double eval(const std::string &expr, const Lookup &lookup) {
    return value(build(expr, lookup, {}, false), expr);
}

Expr parse(const std::string &expr, const Lookup &lookup) { return build(expr, lookup, {}, true); }

std::string text(const Expr &e) {
    std::string out;
    print(e, e.root, out);
    return out;
}

std::vector<std::pair<int, std::string>> expand(const std::vector<std::pair<int, std::string>> &lines, int &errLine, double tempC,
                                                const std::map<std::string, double> &overrides) {
    temperC = tempC;
    return Expander(errLine, overrides).run(lines);
}

}}  // namespace acutesim::numparam
