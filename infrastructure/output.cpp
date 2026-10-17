// Output selection and formatting (see engine_api/output.h).
#include "engine_api/output.h"
#include "netlist/numparam.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <functional>
#include <map>
#include <ostream>
#include <regex>
#include <stdexcept>

namespace acutesim {
using compute::orchestration::AnalysisTypeDTO;
using compute::orchestration::SimulationResponseDTO;
using cplx = std::complex<double>;

namespace {

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string squeeze(std::string s) {
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }), s.end());
    return s;
}

// One expression, evaluated at every point of the response.
class Selector {
public:
    explicit Selector(const SimulationResponseDTO& r) : r_(r), ac_(r.analysisType == AnalysisTypeDTO::AC) {
        for (size_t i = 0; i < r.nodeNames.size(); ++i) if (!r.nodeNames[i].empty()) node_.emplace(r.nodeNames[i], i);
        for (size_t i = 0; i < r.branchNames.size(); ++i) branch_.emplace(r.branchNames[i], i);
    }

    OutputTable::Column column(const std::string& spec) const {
        OutputTable::Column c;
        c.name = lower(squeeze(spec));
        static const std::regex wrap(R"(^(db|mag|ph|phase|real|imag|vdb|vm|vp|vr|vi|idb|im|ip|ir|ii)\((.*)\)$)");
        std::smatch m;
        std::string fn, inner = c.name;
        if (std::regex_match(c.name, m, wrap)) {
            fn = m[1];
            inner = m[2];
            if (fn[0] == 'v' || fn == "idb" || fn == "im" || fn == "ip" || fn == "ir" || fn == "ii") {  // vdb(out) = db(v(out))
                inner = (fn[0] == 'v' ? "v(" : "i(") + inner + ")";
                fn = fn == "vdb" || fn == "idb" ? "db" : fn == "vm" || fn == "im" ? "mag" : fn == "vp" || fn == "ip" ? "ph"
                   : fn == "vr" || fn == "ir" ? "real" : "imag";
            }
        }
        const numparam::Expr e = numparam::parse(inner, [](const std::string&, double&) { return false; });
        std::vector<std::function<cplx(size_t)>> refs;
        for (const auto& s : e.refs) refs.push_back(ref(s));
        c.isComplex = ac_ && fn.empty();
        c.type = fn.empty() && e.n[e.root].op == numparam::Expr::Ref ? (e.refs[0].kind == 'I' ? "current" : "voltage") : "notype";
        for (size_t k = 0; k < r_.waveformPoints.size(); ++k) {
            const cplx v = ac_ ? complexEval(e, refs, k, e.root) : cplx(realEval(e, refs, k), 0.0);
            c.values.push_back(apply(fn, v));
        }
        return c;
    }

private:
    const SimulationResponseDTO& r_;
    bool ac_;
    std::map<std::string, size_t> node_, branch_;

    static cplx apply(const std::string& fn, cplx v) {
        if (fn.empty()) return v;
        if (fn == "db") return 20.0 * std::log10(std::abs(v));
        if (fn == "mag") return std::abs(v);
        if (fn == "ph" || fn == "phase") return std::arg(v);  // radians, as ngspice
        if (fn == "real") return v.real();
        return v.imag();
    }

    std::function<cplx(size_t)> ref(const numparam::Expr::Sym& s) const {
        auto at = [this](size_t k, size_t i, bool branchPart) {
            const auto& p = r_.waveformPoints[k];
            const auto& re = branchPart ? p.branchCurrents : p.nodeVoltages;
            const auto& im = branchPart ? p.branchCurrentsImag : p.nodeVoltagesImag;
            return cplx(i < re.size() ? re[i] : 0.0, i < im.size() ? im[i] : 0.0);
        };
        if (s.kind == 'I') {
            const auto it = branch_.find(lower(s.a));
            if (it == branch_.end())
                throw std::runtime_error("i(" + lower(s.a) + "): only the current of a V, L, E, H or B voltage source is available "
                                         "(device terminal currents are not supported yet)");
            return [=](size_t k) { return at(k, it->second, true); };
        }
        auto node = [this](const std::string& n) -> long {  // -1: ground
            const std::string l = lower(n);
            if (l.empty() || l == "0" || l == "gnd") return -1;
            const auto it = node_.find(l);
            if (it == node_.end()) throw std::runtime_error("v(" + l + "): unknown node");
            return static_cast<long>(it->second);
        };
        const long a = node(s.a), b = node(s.b);
        return [=](size_t k) { return (a < 0 ? cplx() : at(k, a, false)) - (b < 0 ? cplx() : at(k, b, false)); };
    }

    struct Env {
        static constexpr bool strict = false;
        double t;
        const std::vector<double>& vals;
        double time() const { return t; }
        Dual<double> ref(int k) const { return Dual<double>(vals[k]); }
    };
    double realEval(const numparam::Expr& e, const std::vector<std::function<cplx(size_t)>>& refs, size_t k) const {
        std::vector<double> vals;
        for (const auto& f : refs) vals.push_back(f(k).real());
        Env env{r_.waveformPoints[k].timeS, vals};
        return numparam::Evaluator<Env>{e, env}(e.root).val;
    }

    // ponytail: .ac expressions are + - * / over constants and v()/i() only; functions of a complex
    // value (abs, exp, ...) are an error rather than a guess.
    cplx complexEval(const numparam::Expr& e, const std::vector<std::function<cplx(size_t)>>& refs, size_t k, int i) const {
        const auto& n = e.n[i];
        auto a = [&](size_t j) { return complexEval(e, refs, k, n.a[j]); };
        switch (n.op) {
            case numparam::Expr::Const: return n.c;
            case numparam::Expr::Ref: return refs[static_cast<size_t>(n.c)](k);
            case numparam::Expr::Neg: return -a(0);
            case numparam::Expr::Add: return a(0) + a(1);
            case numparam::Expr::Sub: return a(0) - a(1);
            case numparam::Expr::Mul: return a(0) * a(1);
            case numparam::Expr::Div: return a(0) / a(1);
            default: throw std::runtime_error("only + - * / over v() and i() are supported on .ac vectors");
        }
    }
};

void value(std::ostream& os, cplx v, bool complex_) {
    char b[80];
    if (complex_) std::snprintf(b, sizeof b, "%.15e,%.15e", v.real(), v.imag());
    else std::snprintf(b, sizeof b, "%.15e", v.real());
    os << b;
}

}  // namespace

std::string analysisName(const SimulationResponseDTO& r) {
    if (r.analysisType == AnalysisTypeDTO::TRANSIENT) return "tran";
    if (r.analysisType == AnalysisTypeDTO::AC) return "ac";
    return r.scaleName.empty() ? "op" : "dc";
}

OutputTable selectOutputs(const SimulationResponseDTO& r, const std::vector<std::string>& specs) {
    OutputTable t;
    const std::string an = analysisName(r);
    t.plotName = an == "tran" ? "Transient Analysis" : an == "ac" ? "AC Analysis" : an == "dc" ? "DC transfer characteristic"
                                                                                               : "Operating Point";
    t.scaleName = r.scaleName;
    t.complexData = an == "ac";
    for (const auto& p : r.waveformPoints) t.scale.push_back(p.timeS);
    std::vector<std::string> list = specs;
    for (const auto& s : specs) if (lower(s) == "all") list.clear();
    if (list.empty()) {
        for (const auto& n : r.nodeNames) if (!n.empty()) list.push_back("v(" + n + ")");
        for (const auto& n : r.branchNames) list.push_back("i(" + n + ")");
    }
    const Selector sel(r);
    for (const auto& s : list) t.columns.push_back(sel.column(s));
    return t;
}

std::vector<SimulationResponseDTO> splitByOuter(const SimulationResponseDTO& r) {
    std::vector<SimulationResponseDTO> out;
    for (const auto& p : r.waveformPoints) {
        if (out.empty() || out.back().waveformPoints.back().outer != p.outer) {
            out.push_back(r);
            out.back().waveformPoints.clear();
        }
        out.back().waveformPoints.push_back(p);
    }
    if (out.empty()) out.push_back(r);
    return out;
}

std::vector<std::string> requestedSpecs(const SimulationResponseDTO& r, char kind) {
    std::vector<std::string> out;
    const std::string an = analysisName(r);
    for (const auto& q : r.outputRequests)
        if (q.kind == kind && (q.analysis.empty() || q.analysis == an)) out.insert(out.end(), q.specs.begin(), q.specs.end());
    return out;
}

void writeTable(std::ostream& os, const OutputTable& t, const std::string& title) {
    char b[96];
    const std::string rule(80, '-');
    os << "                                     " << title << "\n                                     " << t.plotName
       << "\n" << rule << "\nIndex   ";
    if (!t.scaleName.empty()) { std::snprintf(b, sizeof b, "%-16s", t.scaleName.c_str()); os << b; }
    for (const auto& c : t.columns) { std::snprintf(b, sizeof b, "%-16s", c.name.c_str()); os << b; }
    os << "\n" << rule << "\n";
    for (size_t k = 0; k < std::max<size_t>(t.scale.size(), 1); ++k) {
        os << k << "\t";
        if (!t.scaleName.empty()) { std::snprintf(b, sizeof b, "%.6e\t", t.scale[k]); os << b; }
        for (const auto& c : t.columns) {
            if (c.isComplex) std::snprintf(b, sizeof b, "%.6e,\t%.6e\t", c.values[k].real(), c.values[k].imag());
            else std::snprintf(b, sizeof b, "%.6e\t", c.values[k].real());
            os << b;
        }
        os << "\n";
    }
    os << "\n";
}

void writeRaw(std::ostream& os, const OutputTable& t, const std::string& title, bool ascii) {
    const size_t points = std::max<size_t>(t.scale.size(), 1);
    const bool scaled = !t.scaleName.empty();
    char date[64];
    const std::time_t now = std::time(nullptr);
    std::strftime(date, sizeof date, "%a %b %d %H:%M:%S  %Y", std::localtime(&now));
    os << "Title: " << title << "\nDate: " << date << "\nCommand: treespice\nPlotname: " << t.plotName
       << "\nFlags: " << (t.complexData ? "complex" : "real") << "\nNo. Variables: " << t.columns.size() + scaled
       << "\nNo. Points: " << points << "\nVariables:\n";
    size_t idx = 0;
    if (scaled) {
        const bool freq = t.scaleName == "frequency";
        os << "\t" << idx++ << "\t" << t.scaleName << "\t" << (freq ? "frequency grid=3" : t.scaleName == "time" ? "time" : "voltage") << "\n";
    }
    for (const auto& c : t.columns) os << "\t" << idx++ << "\t" << c.name << "\t" << c.type << "\n";
    os << (ascii ? "Values:\n" : "Binary:\n");
    for (size_t k = 0; k < points; ++k) {
        std::vector<cplx> row;
        if (scaled) row.push_back(t.scale[k]);
        for (const auto& c : t.columns) row.push_back(c.values[k]);
        if (ascii) {
            for (size_t j = 0; j < row.size(); ++j) {
                os << (j == 0 ? " " + std::to_string(k) : std::string()) << "\t";
                value(os, row[j], t.complexData);
                os << "\n";
            }
            os << "\n";
        } else {
            for (const cplx& v : row) {
                const double d[2] = {v.real(), v.imag()};
                os.write(reinterpret_cast<const char*>(d), static_cast<std::streamsize>(t.complexData ? 16 : 8));
            }
        }
    }
}

}  // namespace acutesim
