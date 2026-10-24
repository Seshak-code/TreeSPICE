/**
 * spice_parser.cpp — Full SPICE3f5 Netlist Parser Implementation
 *
 * Converts standard SPICE netlists into the TensorNetlist hierarchy.
 * This replaces the rudimentary deserialiseRequest() in engine_impl.cpp.
 */

#include "spice_parser.h"
#include "numparam.h"
#include "../osdi/osdi_device.h"
#include "../physics/physics_constants.h"
#include "../physics/device_temp.h"
#include "../physics/xspice_analog.h"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <cctype>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace acutesim {

// ============================================================================
// Utility Helpers
// ============================================================================

std::string SPICEParser::toUpper(const std::string& s) {
    std::string r = s;
    for (auto& c : r) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return r;
}

std::vector<std::string> SPICEParser::tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string tok;

    // Handle parenthesized waveform args: PULSE(v1 v2 td ...) → flatten
    std::string flat = line;
    for (auto& c : flat) {
        if (c == '(' || c == ')' || c == ',' || c == '=') c = ' ';
    }

    std::istringstream flatStream(flat);
    while (flatStream >> tok) tokens.push_back(tok);
    return tokens;
}

// numparam's generated names, NAME|3 (a specialized subcircuit) and SUB|MODEL (a model
// hoisted out of one), as the user wrote them: NAME and MODEL.
static std::string userNames(std::string s) {
    auto inName = [](char c) { return !std::isspace((unsigned char)c) && !std::strchr(":,()='\"#.", c); };
    for (size_t bar; (bar = s.find('|')) != std::string::npos;) {
        size_t b = bar, e = bar;
        while (b > 0 && inName(s[b - 1])) --b;
        while (e < s.size() && inName(s[e])) ++e;
        const std::string w = s.substr(b, e - b), last = w.substr(w.rfind('|') + 1);
        const bool specialized = !last.empty() && std::all_of(last.begin(), last.end(), [](char c) { return std::isdigit((unsigned char)c); });
        s.replace(b, e - b, specialized ? w.substr(0, w.find('|')) : last);
    }
    return s;
}

// Split at top-level blanks: v(a, b)*2 and 'x + 1' are one word each.
static std::vector<std::string> splitTop(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    int depth = 0;
    bool quoted = false;
    for (size_t i = 0; i <= s.size(); ++i) {
        const char c = i < s.size() ? s[i] : ' ';
        if (c == '\'') quoted = !quoted;
        if (!quoted) depth += (c == '(') - (c == ')');
        if (std::isspace(static_cast<unsigned char>(c)) && depth <= 0 && !quoted) {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else cur += c;
    }
    return out;
}

// .step [dec|oct|lin] param NAME | temp | V/I/R-element  list v1 v2 ...  |  start stop incr (points per decade/octave for dec/oct)
static SPICEParseResult::Step parseStep(const std::vector<std::string>& w) {
    auto low = [](std::string x) { for (auto& c : x) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return x; };
    auto num = [](std::string x) {
        if (x.size() >= 2 && (x.front() == '{' || x.front() == '\'')) x = x.substr(1, x.size() - 2);
        try { return numparam::eval(x, [](const std::string&, double&) { return false; }); }
        catch (const std::exception&) { throw std::runtime_error(".step: '" + x + "' is not a number"); }
    };
    size_t i = 0;
    std::string mode = "lin";
    if (i < w.size() && (low(w[i]) == "dec" || low(w[i]) == "oct" || low(w[i]) == "lin")) mode = low(w[i++]);
    if (i >= w.size()) throw std::runtime_error(".step needs param NAME, temp or a source/resistor name");
    SPICEParseResult::Step st;
    const std::string what = low(w[i++]);
    if (what == "param") {
        if (i >= w.size()) throw std::runtime_error(".step param needs a name");
        st.kind = 'P';
        st.name = w[i++];
    } else if (what == "temp") { st.kind = 'T'; st.name = "TEMP"; }
    else {
        if (what[0] != 'v' && what[0] != 'i' && what[0] != 'r')
            throw std::runtime_error(".step sweeps a .param, temp, or a V, I or R element, not '" + what + "' (model parameters are not supported yet)");
        st.kind = 'E';
        st.name = what;
    }
    for (auto& c : st.name) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (i < w.size() && low(w[i]) == "list") {
        for (++i; i < w.size(); ++i) st.values.push_back(num(w[i]));
        if (st.values.empty()) throw std::runtime_error(".step list needs at least one value");
    } else if (w.size() - i == 3) {
        const double a = num(w[i]), b = num(w[i + 1]), c = num(w[i + 2]);
        if (mode == "lin") {
            const double span = c != 0.0 ? (b - a) / c : -1.0;
            if (!(span >= -1e-9)) throw std::runtime_error(".step increment must be non-zero and point from start to stop");
            for (long k = 0, n = static_cast<long>(std::floor(span + 1e-9)); k <= n; ++k) st.values.push_back(a + static_cast<double>(k) * c);
        } else {
            if (!(a > 0 && b >= a && c >= 1)) throw std::runtime_error(".step " + mode + " needs 0 < start <= stop and points >= 1");
            const double base = mode == "dec" ? 10.0 : 2.0;
            for (long k = 0; a * std::pow(base, static_cast<double>(k) / std::round(c)) <= b * (1 + 1e-9); ++k)
                st.values.push_back(a * std::pow(base, static_cast<double>(k) / std::round(c)));
        }
    } else throw std::runtime_error(".step expects list v1 v2 ... or start stop incr");
    return st;
}

double SPICEParser::parseValue(const std::string& raw) const {
    // A bare .param name (numparam::expand replaced {expressions} and name=param, and
    // passes the evaluated global parameters as ".param NAME=value" lines)
    std::string s = raw;
    if (auto p = params_.find(toUpper(s)); p != params_.end()) return p->second;

    size_t end = 0;
    double v = 0.0;
    try {
        v = std::stod(s, &end);
    } catch (...) {
        // Was "return 0.0": a typo or {expr} silently became 0 (e.g. a 0-ohm R)
        throw std::runtime_error("'" + raw + "' is not a number or a known .param");
    }

    if (end < s.size()) {
        std::string suffix = s.substr(end);
        // a scale factor plus unit letters only: "0.5*R1" used to read as 0.5
        for (char ch : suffix)
            if (!std::isalpha(static_cast<unsigned char>(ch)))
                throw std::runtime_error("'" + raw + "' is not a number or a known .param");
        std::string sfxUpper = toUpper(suffix);

        if (sfxUpper.rfind("MEG", 0) == 0) v *= 1e6;
        else if (sfxUpper.rfind("MIL", 0) == 0) v *= 25.4e-6; // mils to meters
        else {
            char c = static_cast<char>(std::tolower(static_cast<unsigned char>(suffix[0])));
            switch (c) {
                case 't': v *= 1e12;  break;
                case 'g': v *= 1e9;   break;
                case 'k': v *= 1e3;   break;
                case 'm': v *= 1e-3;  break;
                case 'u': v *= 1e-6;  break;
                case 'n': v *= 1e-9;  break;
                case 'p': v *= 1e-12; break;
                case 'f': v *= 1e-15; break;
                case 'a': v *= 1e-18; break;
                default: break;  // unit letters (V, A, s, ohm) are ignored, as in SPICE
            }
        }
    }
    return v;
}

bool SPICEParser::isValue(const std::string& s) const {
    try { parseValue(s); return true; } catch (...) { return false; }
}

// A card with too few fields used to be silently dropped from the circuit.
void SPICEParser::requireFields(const std::vector<std::string>& tok, size_t n) {
    if (tok.size() < n)
        throw std::runtime_error(tok[0] + ": expected at least " +
                                 std::to_string(n - 1) + " fields after the name");
}

// R, L, I, G, F and J take m= (parallel copies) at the end of the card; C, D, Q and M parse their own.
double SPICEParser::takeMultiplier(std::vector<std::string>& tok) const {
    double m = 1.0;
    for (size_t i = 3; i + 1 < tok.size();) {
        if (toUpper(tok[i]) != "M") { ++i; continue; }
        m = parseValue(tok[i + 1]);
        if (!(m > 0.0)) throw std::runtime_error(tok[0] + ": M must be > 0");
        tok.erase(tok.begin() + i, tok.begin() + i + 2);
    }
    return m;
}

// Instance fields after a D/Q model name: [area] [AREA=a] [M=m]. AREA and M both
// scale the device (currents, capacitances; resistances divide). OFF and IC=
// steer which operating point SPICE finds (ngspice holds OFF junctions at 0 V
// through MODEINITFIX) and the UIC start: errors until they are modelled, not dropped.
double SPICEParser::instanceArea(const std::vector<std::string>& tok, size_t from, InstTemp* t, bool* off,
                                 double* ic, int nIc) const {
    double area = 1.0;
    for (size_t i = from; i < tok.size(); ++i) {
        const std::string k = toUpper(tok[i]);
        if (i == from && isValue(tok[i])) area *= parseValue(tok[i]);
        else if (t && (k == "TEMP" || k == "DTEMP") && i + 1 < tok.size() && isValue(tok[i + 1])) (k == "TEMP" ? t->temp : t->dtemp) = parseValue(tok[++i]);
        else if ((k == "AREA" || k == "M") && i + 1 < tok.size() && isValue(tok[i + 1])) area *= parseValue(tok[++i]);
        else if (k == "OFF" && off) *off = true;
        else if (k == "IC" && ic) {
            int n = 0;
            while (n < nIc && i + 1 < tok.size() && isValue(tok[i + 1])) ic[n++] = parseValue(tok[++i]);
            if (n == 0) throw std::runtime_error(tok[0] + ": IC needs a value");
        }
        else if (k == "OFF" || k == "IC") throw std::runtime_error(tok[0] + ": " + k + " is not supported yet");
        else throw std::runtime_error(tok[0] + ": parameter " + k + " is not supported yet");
    }
    if (!(area > 0.0)) throw std::runtime_error(tok[0] + ": AREA and M must be > 0");
    return area;
}

int SPICEParser::nodeIndex(const std::string& name) {
    std::string upper = toUpper(name);
    // Ground is "0" or "GND", as in ngspice, and global (also inside subcircuit
    // bodies). VSS/GROUND used to be ground too: a cell's VSS port became global
    // ground instead of a port, which shifted every later local node of the cell.
    if (upper == "0" || upper == "GND")
        return 0;

    // Inside a .subckt body node names are local to that definition
    auto& map  = inSubckt_ ? subcktNodeMap_ : nodeMap_;
    auto& next = inSubckt_ ? nextLocalNode_ : nextNode_;
    auto it = map.find(upper);
    if (it != map.end()) {
        if (!noAnalogRef_) (inSubckt_ ? analogLocal_[subcktName_] : analogTop_).insert(it->second);
        return it->second;
    }
    if (inSubckt_ && globals_.count(upper)) subcktGlobals_[subcktName_][next] = upper;  // see expandInstance
    map[upper] = next;
    if (!noAnalogRef_) (inSubckt_ ? analogLocal_[subcktName_] : analogTop_).insert(next);
    return next++;
}

// ============================================================================
// Line Pre-Processing (continuation, comments, blank)
// ============================================================================

std::vector<std::pair<int, std::string>> SPICEParser::preprocessLines(
    const std::string& text)
{
    // Split into raw lines, merge continuation lines ('+' prefix)
    std::vector<std::pair<int, std::string>> raw;
    {
        std::istringstream stream(text);
        std::string line;
        int lineNum = 0;
        while (std::getline(stream, line)) {
            ++lineNum;
            raw.push_back({lineNum, line});
        }
    }

    std::vector<std::pair<int, std::string>> merged;
    for (size_t i = 0; i < raw.size(); ++i) {
        auto& [num, line] = raw[i];

        // Strip inline comment (';')
        auto semiPos = line.find(';');
        if (semiPos != std::string::npos)
            line = line.substr(0, semiPos);

        // Strip trailing whitespace
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back())))
            line.pop_back();
        // and leading whitespace (PDK files indent their "+" continuation lines)
        line.erase(0, line.find_first_not_of(" \t"));

        if (line.empty()) continue;

        // Comment line
        if (line[0] == '*') continue;

        // Continuation line
        if (line[0] == '+' && !merged.empty()) {
            merged.back().second += " " + line.substr(1);
            continue;
        }

        merged.push_back({num, line});
    }
    return merged;
}

// .include <file> splices a file; .lib <file> <section> splices the lines between
// `.lib <section>` and `.endl` of <file> (PDK corners: .lib models.lib tt). Paths
// are relative to the including file. Spliced lines carry the including card's
// line number, which is also reported (errLine) when a file or section is missing.
// ngspice inpcom replace_freq: E/G n1 [n2] FREQ {expr} [=] [MAG|DB] [RAD|DEG] [R_I] values becomes an xfer
// code model (A_GEN<name>) driven by the input node, v(a[, b]), or a B source (B_GEN<name>) for any other
// expression, and .model GEN_MODEL<name> xfer table = [values]; <name> drops the type letter, as ngspice. A line
// the rewrite does not take (ngspice's: a value word over 3 characters in place of a keyword ...) stays as is.
static std::vector<std::pair<int, std::string>> replaceFreq(const std::vector<std::pair<int, std::string>>& lines) {
    std::vector<std::pair<int, std::string>> out;
    auto ws = [](char c) { return c == ' ' || c == '\t'; };
    auto low = [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); };
    for (const auto& [num, line] : lines) {
        const char t = low(line.empty() ? ' ' : line[0]);
        auto keep = [&]() { out.push_back({num, line}); };
        if (t != 'e' && t != 'g') { keep(); continue; }
        const std::string l = [&] { std::string s; for (char c : line) s += low(c); return s; }();
        const size_t n = l.size();
        auto skipWs = [&](size_t i) { while (i < n && ws(l[i])) ++i; return i; };
        auto skipNon = [&](size_t i) { while (i < n && !ws(l[i])) ++i; return i; };
        const size_t e = 1, eE = skipNon(e), n1 = skipWs(eE), n1E = skipNon(n1);
        const size_t fq = l.find("freq", n1E);
        if (fq == std::string::npos || !ws(l[fq - 1]) || fq + 4 >= n || !ws(l[fq + 4])) { keep(); continue; }
        size_t n2 = skipWs(n1E), n2E = 0;
        if (n2 == fq) n2 = std::string::npos;
        else {
            n2E = skipNon(n2);
            if (fq != skipWs(n2E)) { keep(); continue; }
        }
        size_t ex = skipWs(fq + 4);
        if (ex >= n || l[ex] != '{') { keep(); continue; }
        ex = skipWs(ex + 1);
        size_t exE = l.find('}', ex);
        if (exE == std::string::npos) { keep(); continue; }
        while (exE > ex && ws(l[exE - 1])) --exE;
        // the input: a plain name, or v(node) / v(n1, n2)
        size_t in = std::string::npos, inE = 0;
        bool diff = false;
        if (ex < exE && !(l[ex] >= '0' && l[ex] <= '9')) {
            size_t k = ex;
            while (k < exE && ((l[k] >= '0' && l[k] <= '9') || (l[k] >= 'a' && l[k] <= 'z') || l[k] == '_')) ++k;
            if (k == exE) { in = ex; inE = exE; }
        }
        if (exE - ex >= 3 && l[ex] == 'v' && l[ex + 1] == '(' && l[exE - 1] == ')') {
            in = ex + 2; inE = exE - 1;
            const size_t cm = l.find(',', in);
            diff = cm != std::string::npos && cm < inE;
        }
        size_t kw = skipWs(exE + 1);
        if (kw < n && l[kw] == '=') kw = skipWs(kw + 1);
        bool db = true, rad = false, ri = false, gotKey, bad = false;
        size_t list;
        do {
            list = kw;
            if (kw < n && l[kw] == '{') ++kw;
            std::string key;
            while (kw < n && !ws(l[kw]) && l[kw] != '}' && key.size() < 3) key += l[kw++];
            if (kw < n && l[kw] == '}') ++kw;
            if (kw >= n || !ws(l[kw])) { bad = true; break; }
            gotKey = true;
            if (key == "mag") db = false;
            else if (key == "db") db = true;
            else if (key == "rad") rad = true;
            else if (key == "deg") rad = false;
            else if (key == "r_i") ri = true;
            else gotKey = false;
            if (gotKey) list = skipWs(kw);
            kw = list;
        } while (gotKey);
        size_t listE = n;
        while (listE > list && ws(l[listE - 1])) --listE;
        if (bad || list >= n || listE <= list + 1) { keep(); continue; }
        auto s = [&](size_t a, size_t b) { return line.substr(a, b - a); };  // the original case
        const std::string name = s(e, eE), pt = t == 'e' ? "v" : "i";
        const std::string outPort = n2 != std::string::npos ? "%" + pt + "d(" + s(n1, n1E) + " " + s(n2, n2E) + ")"
                                                            : "%" + pt + "(" + s(n1, n1E) + ")";
        if (in != std::string::npos) {
            const std::string inPort = diff ? "%vd(" + s(in, inE) + ")" : s(in, inE);
            out.push_back({num, "a_gen_" + name + " " + inPort + " " + outPort + " gen_model_" + name});
        } else {
            out.push_back({num, "b_gen_" + name + " gen_node_" + name + " 0 v=" + s(ex, exE)});
            out.push_back({num, "a_gen_" + name + " gen_node_" + name + " " + outPort + " gen_model_" + name});
        }
        out.push_back({num, ".model gen_model_" + name + " xfer " + (ri ? "r_i=true" : rad ? "rad=true" : !db ? "db=false" : "") +
                                " table = [" + s(list, listE) + "]"});
    }
    return out;
}

std::vector<std::pair<int, std::string>> SPICEParser::expandIncludes(
    const std::vector<std::pair<int, std::string>>& lines, const std::string& dir, int depth, int& errLine)
{
    if (depth > 16) throw std::runtime_error(".include/.lib nested deeper than 16 files (a cycle?)");
    auto unquote = [](std::string s) {
        if (s.size() >= 2 && (s[0] == '"' || s[0] == '\'') && s.back() == s[0]) s = s.substr(1, s.size() - 2);
        return s;
    };
    auto fields = [&](const std::string& l) {  // first four fields (a quoted one may hold spaces), upper-case keyword
        std::vector<std::string> f(4);
        size_t i = 0;
        for (auto& x : f) {
            while (i < l.size() && std::isspace(static_cast<unsigned char>(l[i]))) ++i;
            if (i == l.size()) break;
            const size_t start = i;
            if (l[i] == '"' || l[i] == '\'') { const size_t e = l.find(l[i], i + 1); i = e == std::string::npos ? l.size() : e + 1; }
            else while (i < l.size() && !std::isspace(static_cast<unsigned char>(l[i]))) ++i;
            x = l.substr(start, i - start);
        }
        f[0] = toUpper(f[0]);
        return f;
    };
    std::vector<std::pair<int, std::string>> out;
    for (const auto& [num, line] : lines) {
        const auto f = fields(line);
        const bool inc = f[0] == ".INCLUDE" || f[0] == ".INC", lib = f[0] == ".LIB" && !f[2].empty();
        if (!inc && !lib) { out.push_back({num, line}); continue; }
        errLine = num;
        if (f[1].empty() || (inc && !f[2].empty()) || !f[3].empty())
            throw std::runtime_error(inc ? ".include takes one file" : ".lib takes a file and a section");
        const std::string file = unquote(f[1]);
        const std::string path = file[0] == '/' || dir.empty() ? file : dir + "/" + file;
        std::ifstream in(path);
        if (!in || !std::filesystem::is_regular_file(path)) throw std::runtime_error(f[0] + ": cannot open '" + path + "'");
        std::stringstream buf;
        buf << in.rdbuf();
        auto body = preprocessLines(buf.str());
        if (lib) {  // keep .lib <section> ... .endl
            const std::string want = toUpper(unquote(f[2]));
            size_t from = 0, to;
            while (from < body.size() && !(fields(body[from].second)[0] == ".LIB" &&
                                           toUpper(unquote(fields(body[from].second)[1])) == want &&
                                           fields(body[from].second)[2].empty()))
                ++from;
            if (from == body.size()) throw std::runtime_error(".lib: no section " + f[2] + " in '" + path + "'");
            for (to = from + 1; to < body.size() && fields(body[to].second)[0] != ".ENDL"; ++to) {}
            if (to == body.size()) throw std::runtime_error(".lib: section " + f[2] + " in '" + path + "' has no .endl");
            body = {body.begin() + from + 1, body.begin() + to};
        }
        // An included file's own .end ends that file, not the netlist (ngspice)
        for (size_t k = 0; k < body.size(); ++k)
            if (fields(body[k].second)[0] == ".END") { body.resize(k); break; }
        for (auto& b : body) b.first = num;
        const size_t slash = path.rfind('/');
        const auto sub = expandIncludes(body, slash == std::string::npos ? "" : path.substr(0, slash), depth + 1, errLine);
        out.insert(out.end(), sub.begin(), sub.end());
    }
    return out;
}

// ============================================================================
// Main Parse Entry Point
// ============================================================================

SPICEParseResult SPICEParser::parse(const std::string& netlistText, const std::string& baseDir, double tempC,
                                    const std::map<std::string, double>& paramOverrides, const SensParam* sens) {
    result_ = SPICEParseResult{};
    baseDir_ = baseDir;
    sens_ = sens;
    tnomC_ = 27.0;
    nodeMap_.clear();
    nextNode_ = 1;
    inSubckt_ = false;
    topInstances_.clear();
    pendingICs_.clear();
    pendingA_.clear();
    subcktICs_.clear();
    globals_.clear();
    subcktGlobals_.clear();
    for (auto& n : pzNames_) n.clear();
    sensOut_.clear();
    sensRef_.clear();
    tfOut_.clear();
    tfRef_.clear();
    noiseOut_.clear();
    noiseRef_.clear();
    subcktChildren_.clear();
    modelCards_.clear();
    inductorIndex_.clear();
    modelNames_.clear();
    params_.clear();
    cardLine_.clear();
    scale_ = 1.0;
    osdiModels_.clear();

    auto lines = preprocessLines(netlistText);
    // .control ... .endc is ngspice's interactive script, not circuit
    for (auto it = lines.begin(); it != lines.end();) {
        const std::string k = toUpper(it->second.substr(0, it->second.find_first_of(" \t")));
        if (k != ".CONTROL") { ++it; continue; }
        auto e = it;
        while (e != lines.end() && toUpper(e->second.substr(0, e->second.find_first_of(" \t"))) != ".ENDC") ++e;
        it = lines.erase(it, e == lines.end() ? e : std::next(e));
    }
    if (lines.empty()) {
        result_.success = false;
        result_.errorMessage = "Empty netlist";
        return result_;
    }

    auto fail = [&](int lineNum, const std::string& what) {
        result_.success = false;
        result_.errorMessage = "Parse error at line " + std::to_string(lineNum) + ": " + userNames(what);
        result_.errorLine = lineNum;
        return result_;
    };

    int incLine = 0;
    try {
        lines = expandIncludes(lines, baseDir, 0, incLine);
        lines = replaceFreq(lines);  // before numparam, as ngspice: the braces are FREQ's
        // .step and .meas are taken out before numparam sees them: their 'expr' and bare words are theirs
        lines.erase(std::remove_if(lines.begin(), lines.end(), [&](const std::pair<int, std::string>& nl) {
            if (nl.first == 1) return false;
            const auto w = splitTop(nl.second);
            if (w.empty()) return false;
            const std::string k = toUpper(w[0]);
            incLine = nl.first;
            if (k == ".STEP") result_.steps.push_back(parseStep({w.begin() + 1, w.end()}));
            else if (k == ".MEAS" || k == ".MEASURE") {
                if (w.size() < 4) throw std::runtime_error(w[0] + " needs an analysis type, a name and a measurement");
                std::string an = w[1];  // tran, dc, ac; another type (Xyce's tran_cont ...) is reported as an error when the analysis ends
                for (auto& c : an) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                result_.outputs.push_back({'m', an, {w.begin() + 2, w.end()}});
            } else return false;
            return true;
        }), lines.end());
        double circuitC = std::isnan(tempC) ? 27.0 : tempC;  // TEMPER: the last .temp / .options temp card, else 27
        for (const auto& [n, l] : lines) {
            const auto t = tokenize(l);
            if (std::isnan(tempC) && !t.empty() && toUpper(t[0]) == ".TEMP" && t.size() > 1) try { circuitC = parseValue(t[1]); } catch (...) {}
            if (std::isnan(tempC) && !t.empty() && (toUpper(t[0]) == ".OPTIONS" || toUpper(t[0]) == ".OPTION"))
                for (size_t i = 1; i + 1 < t.size(); ++i) if (toUpper(t[i]) == "TEMP") try { circuitC = parseValue(t[i + 1]); } catch (...) {}
        }
        lines = numparam::expand(lines, incLine, circuitC, paramOverrides);  // .param expressions, subckt parameters, .if
    } catch (const std::exception& e) { return fail(incLine, e.what()); }

    // .param values are global in SPICE: resolve them before any card uses one
    for (const auto& [lineNum, line] : lines) {
        if (lineNum == 1) continue;
        auto tok = tokenize(line);
        if (!tok.empty() && toUpper(tok[0]) == ".GLOBAL")  // global in every .subckt, wherever declared
            for (size_t i = 1; i < tok.size(); ++i) globals_.insert(toUpper(tok[i]));
        if (!tok.empty() && toUpper(tok[0]) == ".MODEL" && tok.size() > 1)  // N lines need every model name
            modelNames_.insert(toUpper(tok[1]));
        if (!tok.empty() && (toUpper(tok[0]) == ".OPTION" || toUpper(tok[0]) == ".OPTIONS"))  // before any M card
            for (size_t i = 1; i + 1 < tok.size(); ++i)
                if (toUpper(tok[i]) == "SCALE") {
                    try { scale_ = parseValue(tok[i + 1]); } catch (const std::exception& e) { return fail(lineNum, e.what()); }
                    if (!(scale_ > 0.0)) return fail(lineNum, ".option scale must be > 0");
                }
        if (!tok.empty() && toUpper(tok[0]) == ".OSDI") {  // load compiled Verilog-A modules up front
            try {
                std::string path = line.substr(line.find_first_of(" \t") == std::string::npos ? line.size() : line.find_first_of(" \t"));
                path.erase(0, path.find_first_not_of(" \t"));
                if (path.size() >= 2 && (path[0] == '"' || path[0] == '\'') && path.back() == path[0]) path = path.substr(1, path.size() - 2);
                else if (tok.size() != 2) throw std::runtime_error(".osdi needs exactly one library path");
                // a relative path resolves against the netlist's directory, as .include does
                if (const auto p = std::filesystem::path(baseDir) / path;
                    !baseDir.empty() && std::filesystem::path(path).is_relative() && std::filesystem::is_regular_file(p))
                    path = p.string();
                osdi::loadLibrary(path);
            } catch (const std::exception& e) { return fail(lineNum, e.what()); }
            continue;
        }
        if (tok.empty() || toUpper(tok[0]) != ".PARAM") continue;
        try { parseParam(tok); }
        catch (const std::exception& e) { return fail(lineNum, e.what()); }
    }

    for (const auto& [lineNum, line] : lines) {
        if (lineNum == 1) continue; // Line 1 is the SPICE title card

        auto tok = tokenize(line);
        if (tok.empty()) continue;

        result_.linesProcessed++;
        std::string keyword = toUpper(tok[0]);

        try {
            // Directives start with '.'
            if (keyword[0] == '.') {
                if (keyword == ".MODEL")  { parseModelCard(tok); cardLine_[".MODEL/" + toUpper(tok.size() > 1 ? tok[1] : "")] = lineNum; }
                else if (keyword == ".OSDI") {}   // loaded in the first pass
                else if (keyword == ".PARAM")  {} // handled in the first pass
                else if (keyword == ".SUBCKT") parseSubcktBegin(tok);
                else if (keyword == ".ENDS")   parseSubcktEnd();
                else if (keyword == ".OP")     result_.hasOP = true;
                else if (keyword == ".OPTION" || keyword == ".OPTIONS") {
                    // temp/tnom would change every device and are not modelled yet (like .temp). Options the
                    // solver does not read are ignored with a warning, as ngspice does for unknown ones.
                    auto& env = result_.netlist.environment;
                    auto positive = [&](const std::string& k, double v) {
                        if (!(v > 0.0)) throw std::runtime_error(".options " + k + " must be > 0");
                        return v;
                    };
                    for (size_t i = 1; i < tok.size(); ++i) {
                        const std::string k = toUpper(tok[i]);
                        const bool kv = i + 1 < tok.size() && isValue(tok[i + 1]);
                        const double v = kv ? parseValue(tok[i + 1]) : 0.0;
                        if (k == "METHOD" && i + 1 < tok.size()) {
                            const std::string m = toUpper(tok[++i]);
                            if (m != "TRAP" && m != "TRAPEZOIDAL" && m != "GEAR") std::cerr << "Warning: .options method=" << m << " is ignored\n";
                            else env.gear = m == "GEAR";
                            continue;
                        }
                        if (k == "HIER") { flatten_ = kv ? v == 0.0 : false; if (kv) ++i; continue; }  // S3: the quotient circuit
                        if (!kv) { if (k != "SCALE" && k != "GMINSTEPS") std::cerr << "Warning: .options " << k << " is ignored\n"; continue; }
                        ++i;
                        if (k == "GMINSTEPS") env.gmin_stepping = v != 0.0;
                        else if (k == "TEMP") result_.tempsC = {v};
                        else if (k == "TNOM") tnomC_ = v;
                        else if (k == "RELTOL") env.reltol = positive(k, v);
                        else if (k == "ABSTOL") env.abstol = positive(k, v);
                        else if (k == "VNTOL") env.vntol = positive(k, v);
                        else if (k == "TRTOL") env.trtol = positive(k, v);
                        else if (k == "GMIN") { if (v < 0.0) throw std::runtime_error(".options gmin must be >= 0"); env.gmin = v; }
                        else if (k == "RSHUNT") env.rshunt = positive(k, v);
                        else if (k == "ITL1") env.itl1 = (int)positive(k, v);
                        else if (k == "ITL2") env.itl2 = (int)positive(k, v);
                        else if (k == "ITL4") env.itl4 = (int)positive(k, v);
                        else if (k == "MAXORD") env.maxord = (int)positive(k, v);
                        else if (k == "BYPASS") env.bypass = v != 0.0;  // (TREESPICE_BYPASS=0 too, at instantiateOsdi)
                        else if (k != "SCALE") std::cerr << "Warning: .options " << k << " is ignored\n";
                    }
                    // ponytail: chgtol has no counterpart in the engine's LTE (an absolute bound scaled by RELTOL and TRTOL): warned, ignored
                }
                else if (keyword == ".TEMP") {  // a list repeats .op/.dc per temperature (ngspice takes one)
                    if (tok.size() < 2) throw std::runtime_error(".temp needs a temperature");
                    result_.tempsC.clear();
                    for (size_t i = 1; i < tok.size(); ++i) result_.tempsC.push_back(parseValue(tok[i]));
                }
                else if (keyword == ".TRAN") {
                    result_.hasTRAN = true;
                    if (tok.size() >= 3) {
                        result_.tranStep = parseValue(tok[1]);
                        result_.tranStop = parseValue(tok[2]);
                        // .tran tstep tstop [tstart [tmax]] [UIC]
                        if (tok.size() >= 4 && isValue(tok[3])) result_.tranStart = parseValue(tok[3]);
                    }
                    for (const auto& t : tok)
                        if (toUpper(t) == "UIC") result_.tranUIC = true;
                }
                else if (keyword == ".IC" || keyword == ".NODESET") {
                    // .ic / .nodeset V(node)=value ...  → tokens: V node value
                    if (tok.size() < 4 || (tok.size() - 1) % 3 != 0)
                        throw std::runtime_error(keyword + " expects V(node)=value entries");
                    for (size_t i = 1; i + 2 < tok.size(); i += 3) {
                        if (toUpper(tok[i]) != "V")
                            throw std::runtime_error(keyword + " supports V(node)=value only");
                        (inSubckt_ ? subcktICs_[subcktName_] : pendingICs_)
                            .push_back({toUpper(tok[i + 1]), parseValue(tok[i + 2]), lineNum, keyword == ".NODESET"});
                    }
                }
                else if (keyword == ".NOISE") {
                    // .noise v(out[,ref]) src dec|oct|lin n fstart fstop [pts_per_summary]
                    size_t sw = 4;
                    while (sw < tok.size() && toUpper(tok[sw]) != "DEC" && toUpper(tok[sw]) != "OCT" &&
                           toUpper(tok[sw]) != "LIN") ++sw;
                    if (toUpper(tok[1]) != "V" || (sw != 4 && sw != 5) || sw + 3 >= tok.size())
                        throw std::runtime_error(".noise expects v(out[,ref]) src dec|oct|lin n fstart fstop");
                    noiseOut_ = toUpper(tok[2]);
                    noiseRef_ = sw == 5 ? toUpper(tok[3]) : "0";
                    result_.noiseSource = toUpper(tok[sw - 1]);
                    result_.noiseSweep  = toUpper(tok[sw]) == "DEC" ? "dec" : toUpper(tok[sw]) == "OCT" ? "oct" : "lin";
                    result_.noisePoints = static_cast<int>(parseValue(tok[sw + 1]));
                    result_.noiseStart  = parseValue(tok[sw + 2]);
                    result_.noiseStop   = parseValue(tok[sw + 3]);
                }
                else if (keyword == ".SENS") {
                    // .sens v(out[,ref]) | i(vsrc) [ac dec|oct|lin n fstart fstop]
                    const std::string kind = tok.size() > 2 ? toUpper(tok[1]) : "";
                    const size_t want = kind == "V" && tok.size() > 3 && toUpper(tok[3]) != "V" && toUpper(tok[3]) != "I" && toUpper(tok[3]) != "AC" ? 4 : 3;
                    if (kind != "V" && kind != "I") throw std::runtime_error(".sens expects v(out[,ref]) or i(vsrc)");
                    if (tok.size() > want && toUpper(tok[want]) == "AC") {
                        const std::string sw = tok.size() == want + 5 ? toUpper(tok[want + 1]) : "";
                        if (sw != "DEC" && sw != "OCT" && sw != "LIN") throw std::runtime_error(".sens ... ac expects dec|oct|lin n fstart fstop");
                        result_.sensSweep = sw == "DEC" ? "dec" : sw == "OCT" ? "oct" : "lin";
                        result_.sensPoints = static_cast<int>(parseValue(tok[want + 2]));
                        result_.sensStart = parseValue(tok[want + 3]);
                        result_.sensStop = parseValue(tok[want + 4]);
                    } else if (tok.size() != want) throw std::runtime_error(".sens supports a single v(out[,ref]) or i(vsrc) output");
                    result_.hasSens = true;
                    if (kind == "I") result_.sensSource = toUpper(tok[2]);
                    else { sensOut_ = toUpper(tok[2]); sensRef_ = want == 4 ? toUpper(tok[3]) : "0"; }
                }
                else if (keyword == ".TF") {
                    // .tf v(out[,ref]) insrc | .tf i(vsrc) insrc
                    const std::string kind = tok.size() > 3 ? toUpper(tok[1]) : "";
                    if (!(kind == "I" ? tok.size() == 4 : kind == "V" && (tok.size() == 4 || tok.size() == 5)))
                        throw std::runtime_error(".tf expects v(out[,ref]) or i(vsrc), then the input source");
                    const char in = static_cast<char>(std::toupper(static_cast<unsigned char>(tok.back()[0])));
                    if (in != 'V' && in != 'I') throw std::runtime_error(".tf: the input must be a V or I source, not '" + tok.back() + "'");
                    result_.hasTF = true;
                    result_.tfSource = toUpper(tok.back());
                    auto lower = [](std::string s) { for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return s; };
                    if (kind == "I") {
                        result_.tfOutSource = toUpper(tok[2]);
                        if (result_.tfOutSource[0] != 'V') throw std::runtime_error(".tf i(" + tok[2] + "): the output current must be a V source's");
                    } else {
                        tfOut_ = toUpper(tok[2]);
                        tfRef_ = tok.size() == 5 ? toUpper(tok[3]) : "0";
                        result_.tfOutLabel = "v(" + lower(tok[2]) + (tok.size() == 5 ? "," + lower(tok[3]) : "") + ")";
                    }
                }
                else if (keyword == ".PZ") {
                    // .pz in+ in- out+ out- cur|vol pol|zer|pz
                    const std::string io = tok.size() == 7 ? toUpper(tok[5]) : "", what = tok.size() == 7 ? toUpper(tok[6]) : "";
                    if ((io != "CUR" && io != "VOL") || (what != "POL" && what != "ZER" && what != "PZ"))
                        throw std::runtime_error(".pz expects in+ in- out+ out- cur|vol pol|zer|pz");
                    for (int k = 0; k < 4; ++k) pzNames_[k] = toUpper(tok[k + 1]);
                    result_.hasPZ = true;
                    result_.pzVolt = io == "VOL";
                    result_.pzPoles = what != "ZER";
                    result_.pzZeros = what != "POL";
                }
                else if (keyword == ".AC") {
                    // .ac dec|oct|lin n fstart fstop
                    const std::string sw = tok.size() == 5 ? toUpper(tok[1]) : "";
                    if (sw != "DEC" && sw != "OCT" && sw != "LIN")
                        throw std::runtime_error(".ac expects dec|oct|lin n fstart fstop");
                    result_.acSweep = sw == "DEC" ? "dec" : sw == "OCT" ? "oct" : "lin";
                    result_.acPoints = static_cast<int>(parseValue(tok[2]));
                    result_.acStart = parseValue(tok[3]);
                    result_.acStop  = parseValue(tok[4]);
                }
                else if (keyword == ".DC") {
                    // .dc SRC start stop step [SRC2 start stop step ...]: the first is the innermost sweep
                    if (tok.size() < 5 || (tok.size() - 1) % 4 != 0)
                        throw std::runtime_error(".dc expects SRC start stop step [SRC2 start stop step ...]");
                    result_.dcSweeps.clear();
                    for (size_t i = 1; i < tok.size(); i += 4) {
                        const std::string name = toUpper(tok[i]);
                        const char c = name[0];
                        if (c != 'V' && c != 'I' && c != 'R' && name != "TEMP")
                            throw std::runtime_error(".dc sweeps a V or I source, a resistor or TEMP only, not '" + tok[i] + "'");
                        result_.dcSweeps.push_back({name, parseValue(tok[i + 1]), parseValue(tok[i + 2]), parseValue(tok[i + 3])});
                    }
                }
                else if (keyword == ".PRINT" || keyword == ".SAVE" || keyword == ".PROBE") {
                    OutputRequest out{keyword == ".PRINT" ? 'p' : 's', "", {}};
                    out.specs = splitTop(line.substr(keyword.size()));
                    if (!out.specs.empty() && std::set<std::string>{"TRAN", "AC", "DC", "OP"}.count(toUpper(out.specs[0]))) {
                        out.analysis = out.specs[0];
                        for (auto& c : out.analysis) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                        out.specs.erase(out.specs.begin());
                    }
                    if (out.specs.empty()) throw std::runtime_error(tok[0] + " needs at least one output");
                    if (!inSubckt_) result_.outputs.push_back(std::move(out));
                }
                else if (keyword == ".END") break;
                // Output-only (.four only prints a Fourier table) and first-pass cards; anything else changes the
                // circuit or analysis (.step, .nodeset, ...) and used to be
                // ignored silently.
                else if (keyword != ".GLOBAL" && keyword != ".PLOT" && keyword != ".MEAS" &&
                         keyword != ".MEASURE" &&
                         keyword != ".WIDTH" && keyword != ".TITLE" && keyword != ".FOUR")
                    throw std::runtime_error(tok[0] + " is not supported yet");
                continue;
            }

            // Choose target block: subcircuit body or global
            TensorBlock& blk = inSubckt_ ? subcktBlock_ : result_.netlist.globalBlock;
            char type = static_cast<char>(std::toupper(static_cast<unsigned char>(keyword[0])));
            mult_ = std::strchr("RLIGFJZ", type) ? takeMultiplier(tok) : 1.0;

            switch (type) {
                case 'R': parseResistor(tok, blk);   break;
                case 'C': parseCapacitor(tok, blk);   break;
                case 'L': parseInductor(tok, blk);    break;
                case 'K': parseMutualInd(tok, blk);   break;
                case 'V': parseVoltSrc(tok, blk);     break;
                case 'I': parseCurrSrc(tok, blk);     break;
                case 'D': parseDiode(tok, blk);       break;
                case 'M': parseMosfet(tok, blk);      break;
                case 'Q': parseBJT(tok, blk);         break;
                case 'J': parseJFET(tok, blk);        break;
                case 'Z': parseJFET(tok, blk); blk.jfets.back().mes = true; break;  // MESFET: d g s model
                case 'U': parseURC(tok, blk); break;
                case 'S': case 'W': parseSwitch(tok, blk); break;
                case 'N': parseOsdiDevice(tok, blk);  break;
                case 'A': parseXspice(line, lineNum); break;
                case 'B': parseBehav(line, tok, blk); break;
                case 'E': nonlinearForm(tok) ? parseBehav(line, tok, blk) : parseVCVS(tok, blk); break;
                case 'G': nonlinearForm(tok) ? parseBehav(line, tok, blk) : parseVCCS(tok, blk); break;
                case 'H': nonlinearForm(tok) ? parseBehav(line, tok, blk) : parseCCVS(tok, blk); break;
                case 'F': nonlinearForm(tok) ? parseBehav(line, tok, blk) : parseCCCS(tok, blk); break;
                case 'T': parseTLine(tok, blk);       break;
                case 'O': parseLossyLine(tok, blk); break;
                case 'X': {
                    // X<name> <nodes...> <subcktName>; expanded after parsing
                    requireFields(tok, 3);
                    PendingInstance pi{toUpper(tok.back()), {}, lineNum, toUpper(tok[0]), {}};
                    noAnalogRef_ = true;  // an X port is analog only if the body uses it so (expandInstance)
                    for (size_t i = 1; i + 1 < tok.size(); ++i) {
                        pi.nodes.push_back(nodeIndex(tok[i]));
                        pi.portNames.push_back(toUpper(tok[i]));
                    }
                    noAnalogRef_ = false;
                    (inSubckt_ ? subcktChildren_[subcktName_] : topInstances_).push_back(pi);
                    break;
                }
                default:
                    // Was a warning: the element was dropped and the rest
                    // of the circuit "converged" without it.
                    throw std::runtime_error("unknown element '" + tok[0] + "'");
            }
            result_.elementsCreated++;
            cardLine_[(inSubckt_ ? subcktName_ : std::string()) + "/" + toUpper(tok[0])] = lineNum;
        } catch (const std::exception& e) {
            return fail(lineNum, e.what());
        }
    }

    if (inSubckt_)
        return fail(lines.back().first, ".subckt " + subcktName_ + " has no .ends");
    // ponytail: the engine's Gear 2 has constant-step coefficients, wrong under LTE step changes, so method=gear
    // with maxord > 1 runs the trapezoidal rule (same solution within LTE); maxord=1 is exact backward Euler.
    if (result_.netlist.environment.gear && result_.netlist.environment.maxord > 1)
        std::cerr << "Warning: .options method=gear (maxord > 1) runs as trapezoidal\n";
    try { resolveControlledSources(result_.netlist.globalBlock); }
    catch (const std::exception& e) { return fail(lines.back().first, e.what()); }


    // Finalize: compute numGlobalNodes from nodeMap (max index)
    result_.nodeIndex = nodeMap_;
    for (const auto& [name, idx] : nodeMap_) {
        if (idx > result_.netlist.numGlobalNodes)
            result_.netlist.numGlobalNodes = idx;
    }

    // .noise output nodes, resolved once every node is known
    if (!noiseOut_.empty()) {
        auto node = [&](const std::string& nm) {
            if (nm == "0" || nm == "GND") return 0;
            auto it = nodeMap_.find(nm);
            return it == nodeMap_.end() ? -1 : it->second;
        };
        result_.noiseOut = node(noiseOut_);
        result_.noiseRef = node(noiseRef_);
        if (result_.noiseOut <= 0 || result_.noiseRef < 0)
            return fail(lines.back().first, ".noise names an unknown output node");
    }

    if (!sensOut_.empty()) {
        auto node = [&](const std::string& nm) {
            if (nm == "0" || nm == "GND") return 0;
            auto it = nodeMap_.find(nm);
            return it == nodeMap_.end() ? -1 : it->second;
        };
        result_.sensOut = node(sensOut_);
        result_.sensRef = node(sensRef_);
        if (result_.sensOut < 0 || result_.sensRef < 0) return fail(lines.back().first, ".sens names an unknown output node");
    }
    if (!tfOut_.empty()) {
        auto node = [&](const std::string& nm) {
            if (nm == "0" || nm == "GND") return 0;
            auto it = nodeMap_.find(nm);
            return it == nodeMap_.end() ? -1 : it->second;
        };
        result_.tfOut = node(tfOut_);
        result_.tfRef = node(tfRef_);
        if (result_.tfOut < 0 || result_.tfRef < 0) return fail(lines.back().first, ".tf names an unknown output node");
    }
    if (result_.hasPZ)
        for (int k = 0; k < 4; ++k) {
            const std::string& nm = pzNames_[k];
            auto it = nodeMap_.find(nm);
            if (nm == "0" || nm == "GND") result_.pzNode[k] = 0;
            else if (it == nodeMap_.end()) return fail(lines.back().first, ".pz names an unknown node '" + nm + "'");
            else result_.pzNode[k] = it->second;
        }

    auto card = [&](const std::string& model) -> const ModelCard* {
        auto it = modelCards_.find(toUpper(model));
        if (it == modelCards_.end()) throw std::runtime_error("model '" + model + "' is not defined");
        return &it->second;
    };
    // A MOSFET's card: its own, or the bin of a family NAME.0, NAME.1, ... whose
    // [LMIN, LMAX] x [WMIN, WMAX] holds the device's L and W. ngspice's rule (probed
    // on ngspice 45): edges inclusive within 1e-9 m, total W (not W/NF), and on a
    // shared edge the card defined last wins (its model list is newest-first).
    auto mosCard = [&](const Mosfet& m) -> const ModelCard* {
        const std::string name = toUpper(m.modelName);
        if (modelCards_.count(name)) return card(m.modelName);
        constexpr double eps = 1e-9;
        const ModelCard* best = nullptr;
        int bestLine = -1;
        for (auto it = modelCards_.lower_bound(name + "."); it != modelCards_.end() &&
             it->first.compare(0, name.size() + 1, name + ".") == 0; ++it) {
            const ModelCard& c = it->second;
            const int line = cardLine_[".MODEL/" + it->first];
            if (line >= bestLine &&
                c.get("LMIN", 0.0) - eps <= m.l && m.l <= c.get("LMAX", INFINITY) + eps &&
                c.get("WMIN", 0.0) - eps <= m.w && m.w <= c.get("WMAX", INFINITY) + eps) { best = &c; bestLine = line; }
        }
        if (best) return best;
        std::ostringstream why;
        why << m.instanceName << ": no bin of model " << m.modelName << " covers L=" << m.l << " W=" << m.w;
        throw std::runtime_error(why.str());
    };
    // RD / RS (or RSH * NRD / NRS) of a level 1 MOSFET are series resistors to internal nodes, added here
    // before the X cards expand (a .subckt body gets local nodes, the top level global ones). ngspice
    // gives RD precedence over RSH, and M parallel devices divide the resistance.
    // The card a device resolves: its own, or a renamed copy (a model of its own) with the .sens override applied
    auto sensed = [&](const ModelCard* c, const std::string& inst) -> const ModelCard* {
        if (!sens_ || !sens_->model || toUpper(inst) != sens_->instance) return c;
        sensCard_ = *c;
        sensCard_.name += "#SENS";
        sensCard_.params[sens_->param] = sens_->value;
        return &sensCard_;
    };
    auto mosSeries = [&](TensorBlock& blk, bool top) {
        const size_t n = blk.mosfets.size();
        for (size_t k = 0; k < n; ++k) {
            Mosfet& m = blk.mosfets[k];
            const ModelCard* c;
            try { c = sensed(mosCard(m), m.instanceName); } catch (const std::exception&) { continue; }  // resolveBlock reports it
            if (c->type != "NMOS" && c->type != "PMOS") continue;
            if (const double lv = c->get("LEVEL", 1.0); lv == 54.0 || lv == 14.0 || lv == 49.0 || lv == 8.0) continue;  // BSIM4 / BSIM3 (OSDI) keep RD, RS inside
            auto add = [&](int Mosfet::* term, const char* tag, double rd, double rs_, double nrx) {
                const double r = rd > 0.0 ? rd : rs_ > 0.0 ? rs_ * (nrx > 0.0 ? nrx : 1.0) : 0.0;
                if (r <= 0.0) return;
                const int inner = top ? std::max(nextNode_++, result_.netlist.numGlobalNodes + 1) : ++blk.numInternalNodes;
                if (top) result_.netlist.numGlobalNodes = std::max(result_.netlist.numGlobalNodes, inner);
                Resistor res;
                res.name = std::string(tag) + "#" + m.instanceName;
                res.nodeTerminal1 = m.*term;
                res.nodeTerminal2 = inner;
                res.resistance_ohms = r / m.extra.mult;
                blk.resistors.push_back(res);
                m.*term = inner;
            };
            add(&Mosfet::drain, "RD", c->get("RD", 0.0), c->get("RSH", 0.0), m.geo.nrd);
            add(&Mosfet::source, "RS", c->get("RS", 0.0), c->get("RSH", 0.0), m.geo.nrs);
        }
    };
    // The same for a JFET's RD and RS (conductance scales with AREA)
    auto jfetSeries = [&](TensorBlock& blk, bool top) {
        for (JFET& j : blk.jfets) {
            const auto it = modelCards_.find(toUpper(j.modelName));
            if (it == modelCards_.end()) continue;  // resolveBlock reports it
            auto add = [&](int JFET::* term, const char* tag, double r) {
                if (r <= 0.0) return;
                const int inner = top ? std::max(nextNode_++, result_.netlist.numGlobalNodes + 1) : ++blk.numInternalNodes;
                if (top) result_.netlist.numGlobalNodes = std::max(result_.netlist.numGlobalNodes, inner);
                Resistor res;
                res.name = std::string(tag) + "#" + j.instanceName;
                res.nodeTerminal1 = j.*term;
                res.nodeTerminal2 = inner;
                res.resistance_ohms = r / j.area;
                blk.resistors.push_back(res);
                j.*term = inner;
            };
            const ModelCard* c = sensed(&it->second, j.instanceName);
            add(&JFET::drain, "RD", c->get("RD", 0.0));
            add(&JFET::source, "RS", c->get("RS", 0.0));
        }
    };
    // A HICUM device (BJT LEVEL 8) without a thermal node gets an internal one, as in ngspice; a VBIC (LEVEL 4, 9)
    // one has it on ground (ngspice ties missing VBIC ports to ground); Gummel-Poon has none
    auto bjtThermal = [&](TensorBlock& blk, bool top) {
        for (BJT& q : blk.bjts) {
            const auto it = modelCards_.find(toUpper(q.modelName));
            if (it == modelCards_.end()) continue;  // resolveBlock reports it
            const double level = it->second.get("LEVEL", 1.0);
            if (level == 4.0 || level == 9.0) { if (q.thermal < 0) q.thermal = 0; continue; }
            if (level != 8.0) {
                if (q.thermal >= 0) throw std::runtime_error(q.instanceName + ": a thermal node needs a HICUM (LEVEL 8) or VBIC (LEVEL 4, 9) model");
                continue;
            }
            if (q.thermal >= 0) continue;
            q.thermal = top ? std::max(nextNode_++, result_.netlist.numGlobalNodes + 1) : ++blk.numInternalNodes;
            if (top) result_.netlist.numGlobalNodes = std::max(result_.netlist.numGlobalNodes, q.thermal);
        }
    };
    auto supported = [&](const ModelCard* c, const std::vector<const char*>& used) {
        for (const auto& [k, v] : c->params) {
            bool ok = v == 0.0 || (k == "LEVEL" && v == 1.0) || (k == "AF" && c->get("KF", 0.0) == 0.0);
            for (const char* u : used) ok = ok || k == u;
            if (!ok) throw std::runtime_error("model " + c->name + ": parameter " + k + " is not supported yet");
        }
    };
    // Magnitudes (>= 0) and scales (> 0): a negative value is a sign error, never a
    // model; it used to be simulated silently (VAF = -5 put 18 V on a 3 V supply).
    auto bounded = [&](const ModelCard* c, std::initializer_list<const char*> nonNeg,
                       std::initializer_list<const char*> pos) {
        for (const char* k : nonNeg)
            if (c->params.count(k) && !(c->params.at(k) >= 0.0))
                throw std::runtime_error("model " + c->name + ": " + k + " must be >= 0");
        for (const char* k : pos)
            if (c->params.count(k) && !(c->params.at(k) > 0.0))
                throw std::runtime_error("model " + c->name + ": " + k + " must be > 0");
    };
    // U lines (ngspice urcsetup): two R ladders, from either end to the middle, each section's resistance and
    // capacitance (to n3) growing by K; with ISPERL, diodes (IS, CJO, RS per section) in place of the capacitors
    auto urcExpand = [&](TensorBlock& blk, bool top) {
        for (const auto& u : blk.urcs) {
            const ModelCard* c = card(u.model);
            if (c->type != "URC") throw std::runtime_error(u.name + ": model " + c->name + " is " + c->type + ", not a URC model");
            supported(c, {"K", "FMAX", "RPERL", "CPERL", "ISPERL", "RSPERL"});
            bounded(c, {"CPERL", "ISPERL", "RSPERL"}, {"FMAX", "RPERL"});
            const double p = c->get("K", 1.5);
            if (!(p > 1.0)) throw std::runtime_error("model " + c->name + ": K must be > 1");
            const double r0 = u.length * c->get("RPERL", 1000.0), c0 = u.length * c->get("CPERL", 1e-12), i0 = u.length * c->get("ISPERL", 0.0);
            int lumps = u.lumps;
            if (lumps == 0) {
                const double wnorm = c->get("FMAX", 1e9) * r0 * c0 * 2.0 * M_PI;
                lumps = wnorm < 35.0 ? 3 : (int)std::max(3.0, std::log(wnorm * ((p - 1) / p) * ((p - 1) / p)) / std::log(p));
            }
            const double r1 = r0 * (p - 1) / (2 * std::pow(p, lumps) - 2);
            const double c1 = c0 * (p - 1) / (std::pow(p, lumps - 1) * (p + 1) - 2);
            const double i1 = i0 * (p - 1) / (std::pow(p, lumps - 1) * (p + 1) - 2);
            const bool diodes = c->params.count("ISPERL") > 0;
            std::string dmodel;
            if (diodes) {
                dmodel = toUpper(u.name) + "#DIODEMOD";
                ModelCard d;
                d.name = dmodel; d.type = "D";
                d.params = {{"CJO", c1}, {"RS", u.length * lumps * c->get("RSPERL", 0.0)}, {"IS", i1}};
                modelCards_[dmodel] = d;
            }
            auto node = [&]() {
                const int n = top ? std::max(nextNode_++, result_.netlist.numGlobalNodes + 1) : ++blk.numInternalNodes;
                if (top) result_.netlist.numGlobalNodes = std::max(result_.netlist.numGlobalNodes, n);
                return n;
            };
            auto shunt = [&](const std::string& tag, int n, double prop) {
                if (diodes) {
                    Diode d;
                    d.instanceName = u.name + "#" + tag; d.anode = n; d.cathode = u.n3; d.modelName = dmodel;
                    d.area = prop; d.areaGiven = true;
                    blk.diodes.push_back(d);
                } else {
                    Capacitor cap{};
                    cap.name = u.name + "#" + tag; cap.nodePlate1 = n; cap.nodePlate2 = u.n3; cap.capacitance_farads = prop * c1;
                    blk.capacitors.push_back(cap);
                }
            };
            auto res = [&](const std::string& tag, int a, int b, double r) {
                Resistor x{};
                x.name = u.name + "#" + tag; x.nodeTerminal1 = a; x.nodeTerminal2 = b; x.resistance_ohms = r;
                blk.resistors.push_back(x);
            };
            int lowl = u.n1, hir = u.n2;
            double prop = 1.0;
            for (int i = 1; i <= lumps; ++i) {
                const int hil = node(), lowr = i == lumps ? hil : node();
                res("RLO" + std::to_string(i), lowl, lowr, prop * r1);
                res("RHI" + std::to_string(i), hil, hir, prop * r1);
                shunt("LO" + std::to_string(i), lowr, prop);
                if (i != lumps) shunt("HI" + std::to_string(i), hil, prop);
                prop *= p;
                lowl = lowr; hir = hil;
            }
        }
        blk.urcs.clear();
    };
    try {
        urcExpand(result_.netlist.globalBlock, true);
        for (auto& [name, blk] : result_.netlist.blockDefinitions) urcExpand(blk, false);
        // O lines: their LTRA model (ngspice ltraset / ltratemp)
        auto ltraResolve = [&](TensorBlock& blk) {
            for (auto& o : blk.lossyLines) {
                const ModelCard* c = card(o.modelName);
                if (c->type != "LTRA") throw std::runtime_error(o.name + ": model " + c->name + " is " + c->type + ", not an LTRA model");
                supported(c, {"R", "L", "G", "C", "LEN", "NOCONTROL", "STEPLIMIT", "NOSTEPLIMIT", "LININTERP", "QUADINTERP",
                              "MIXEDINTERP", "TRUNCNR", "TRUNCDONTCUT", "COMPACTREL", "COMPACTABS"});
                bounded(c, {"R", "L", "G", "C", "LEN", "COMPACTREL", "COMPACTABS"}, {});
                auto& m = o.m;
                m.R = c->get("R", 0.0); m.L = c->get("L", 0.0); m.G = c->get("G", 0.0); m.C = c->get("C", 0.0); m.len = c->get("LEN", 0.0);
                m.interp = c->params.count("LININTERP") ? ltra::Interp::LIN : c->params.count("MIXEDINTERP") ? ltra::Interp::MIXED : ltra::Interp::QUAD;
                m.stepLimit = !c->params.count("NOSTEPLIMIT");
                m.truncDontCut = c->params.count("TRUNCDONTCUT") > 0;
                const auto& env = result_.netlist.environment;
                const double sr = c->get("COMPACTREL", 0.0), sa = c->get("COMPACTABS", 0.0);
                ltra::setup(m, c->name, sr != 0.0 ? sr : env.reltol, sa != 0.0 ? sa : env.abstol);
            }
        };
        ltraResolve(result_.netlist.globalBlock);
        for (auto& [name, blk] : result_.netlist.blockDefinitions) ltraResolve(blk);
        mosSeries(result_.netlist.globalBlock, true);
        jfetSeries(result_.netlist.globalBlock, true);
        bjtThermal(result_.netlist.globalBlock, true);
        for (auto& [name, blk] : result_.netlist.blockDefinitions) { mosSeries(blk, false); jfetSeries(blk, false); bjtThermal(blk, false); }
    } catch (const std::exception& e) { return fail(lines.back().first, e.what()); }

    // X cards expanded (recursively) now that every .subckt is defined. Without flattening, the hierarchy as written
    // (netlist.hier) instead, each definition's X cards checked once (S3: O(definitions)); the flat path does not
    // keep it (a top-level array of cells would hold its instances twice)
    if (!flatten_) {
        auto& h = result_.netlist.hier;
        for (const auto& pi : topInstances_) h.top.push_back({pi.subckt, pi.name, pi.nodes});
        for (const auto& [def, kids] : subcktChildren_)
            for (const auto& c : kids) h.children[def].push_back({c.subckt, c.name, c.nodes});
        h.nodeNames = subcktNames_;
        for (const auto& [def, ics] : subcktICs_)
            for (const auto& ic : ics) h.seeds[def].push_back({subcktNames_[def].at(ic.node), ic.value, ic.nodeset});
    }
    if (flatten_) {
        for (const auto& pi : topInstances_) {
            try { expandInstance(pi, pi.nodes, 0, "", pi.portNames, pi.line); }
            catch (const std::exception& e) { return fail(pi.line, e.what()); }
        }
    } else {
        std::map<std::string, int> seen;  // 1: being checked (a cycle back to it is a recursive .subckt), 2: done
        std::function<void(const PendingInstance&, int)> check = [&](const PendingInstance& pi, int depth) {
            auto it = result_.netlist.blockDefinitions.find(pi.subckt);
            if (it == result_.netlist.blockDefinitions.end()) throw std::runtime_error("unknown subcircuit '" + pi.subckt + "'");
            if (pi.nodes.size() != it->second.pinNames.size())
                throw std::runtime_error(pi.subckt + " has " + std::to_string(it->second.pinNames.size()) +
                                         " pins but the instance connects " + std::to_string(pi.nodes.size()));
            int& st = seen[pi.subckt];
            if (st == 2) return;
            if (st == 1 || depth > 64) throw std::runtime_error("subcircuit nesting deeper than 64 (recursive .subckt?)");
            st = 1;
            if (!subcktA_[pi.subckt].empty())
                throw std::runtime_error(pi.subckt + ": XSPICE devices in a subcircuit need the flattened netlist");
            if (!subcktGlobals_[pi.subckt].empty())
                throw std::runtime_error(pi.subckt + ": .global nets inside a subcircuit need the flattened netlist");
            for (const auto& c : subcktChildren_[pi.subckt]) check(c, depth + 1);
            st = 2;
        };
        for (const auto& pi : topInstances_) {
            try { check(pi, 0); }
            catch (const std::exception& e) { return fail(pi.line, e.what()); }
        }
    }
    // XSPICE A devices, a body's per instance (in netlist order: a node folds its drivers in that order)
    {
        std::stable_sort(pendingA_.begin(), pendingA_.end(), [](const PendingA& a, const PendingA& b) { return a.line < b.line; });
        int aLine = 0;
        try { resolveXspice(aLine); } catch (const std::exception& e) { return fail(aLine ? aLine : lines.back().first, e.what()); }
    }

    // .ic / .nodeset may name a node defined later in the file, or a subcircuit internal (x1.n)
    for (const auto& ic : pendingICs_) {
        auto it = nodeMap_.find(ic.node);
        const bool top = it != nodeMap_.end();
        if (!top && !flatten_ && ic.node.find('.') != std::string::npos)
            return fail(ic.line, (ic.nodeset ? ".nodeset" : ".ic") + std::string(" on subcircuit node '") + ic.node +
                                 "' needs the flattened netlist");
        if (!top) it = result_.nodeIndex.find(ic.node);
        if (!top && it == result_.nodeIndex.end())
            return fail(ic.line, (ic.nodeset ? ".nodeset" : ".ic") + std::string(" names unknown node '") + ic.node + "'");
        (ic.nodeset ? result_.netlist.nodeSets : result_.netlist.nodeICs).push_back({it->second, ic.value});
    }

    for (auto* v : {&result_.netlist.nodeICs, &result_.netlist.nodeSets}) {  // one value per node: the last given (a top-level card beats a body's)
        std::map<int, double> last;
        for (const auto& [n, val] : *v) last[n] = val;
        v->assign(last.begin(), last.end());
    }

    result_.netlist.nodeNames.assign(result_.netlist.numGlobalNodes + 1, "");
    for (const auto* names : {&nodeMap_, &result_.nodeIndex})
        for (const auto& [name, idx] : *names)
            if (idx >= 0 && idx < (int)result_.netlist.nodeNames.size()) result_.netlist.nodeNames[idx] = name;

    ambientK_ = (std::isnan(tempC) ? (result_.tempsC.empty() ? 27.0 : result_.tempsC[0]) : tempC) + 273.15;
    result_.netlist.environment.ambient_temp_K = ambientK_;
    result_.netlist.environment.temp_count = std::max<int>(1, (int)result_.tempsC.size());
    const double tnomK = tnomC_ + 273.15;
    // Every temperature-dependent value below is resolved at ambientK_ (instance TEMP/DTEMP shift it), as ngspice's
    // *temp.c from the model's TNOM (.options tnom, 27 C); a netlist at TNOM keeps its cards exactly.

    // Resolve .model cards now that the whole file is read (SPICE allows .model
    // after the instance; resolving at the instance made a late PNP an NPN).
    // A card's missing parameters take SPICE defaults; a device with no card
    // keeps the engine defaults. PULSE TR/TF of 0 take TSTEP, as in SPICE.
    // An undefined model is an error (it used to fall back to engine defaults silently)
    // A parameter the engine does not model is an error, never silently dropped.
    // 0 is accepted for any key (SPICE: off / infinite), AF only without KF.
    // ponytail: a 0 that is not "off" (e.g. TNOM=0) is still accepted.
    // A top-level device's real parameters for .sens: what its card gives, plus instance W/L as given
    auto sensList = [&](const TensorBlock& blk, const std::string& inst, const ModelCard& c,
                        const std::map<std::string, double>* given = nullptr) {
        if (&blk != &result_.netlist.globalBlock) return;
        auto add = [&](const std::string& k, double v, bool model) {
            if (v == 0.0) ++result_.sensZeroParams;
            else result_.sensParams.push_back({toUpper(inst), k, model, v});
        };
        // ponytail: selectors are recognised by name (LEVEL, VERSION, *MOD, ...); a missed one differentiates to 0
        for (const auto& [k, v] : c.params)
            if (k != "LEVEL" && k != "VERSION" && k != "TYPE" && k != "LMIN" && k != "LMAX" && k != "WMIN" && k != "WMAX" &&
                k != "ACM" && k != "PARAMCHK" && k != "BINUNIT" && !(k.size() > 3 && k.compare(k.size() - 3, 3, "MOD") == 0))
                add(k, v, true);
        if (given) for (const char* k : {"W", "L"}) if (const auto it = given->find(k); it != given->end()) add(k, it->second, false);
    };
    // The card's type must fit the device (a Q on an NMOS card reported "VTO is not supported")
    auto kind = [&](const ModelCard* c, const std::string& inst, std::initializer_list<const char*> types) {
        for (const char* t : types) if (c->type == t) return sensed(c, inst);
        throw std::runtime_error(inst + ": model " + c->name + " is " + c->type + ", not a " +
                                 std::string(*types.begin()) + "/" + *(types.begin() + 1) + " model");
    };
    // VDMOS (ngspice vdmos*.c) through OSDI: tests/models/vdmos_ng.va, a port of ngspice 47's model, as vdmos.osdi
    // from TREESPICE_OSDI_PATH. M d g s model, or M d g s tj tc model THERMAL (self-heating, tj and tc in deg C).
    auto vdmosOsdi = [&](const Mosfet& m, const ModelCard& c) {
        static const std::map<std::string, std::string> alias = {{"VTH0", "VTO"}, {"VTOTC", "TCVTH"}, {"BEX", "MU"}, {"M", "MJ"}};
        static const std::set<std::string> soa = {"VGS_MAX", "VGD_MAX", "VDS_MAX", "VGSR_MAX", "VGDR_MAX", "PD_MAX",
                                                  "ID_MAX", "IDR_MAX", "TE_MAX", "RTH_EXT", "DERATING"};  // warnings only
        static const std::set<std::string> known = {"VTO", "KP", "PHI", "LAMBDA", "THETA", "RD", "RS", "RG", "TNOM",
            "KF", "AF", "RQ", "VQ", "MTRIODE", "TCVTH", "MU", "TEXP0", "TEXP1", "TRD1", "TRD2", "TRG1", "TRG2", "TRS1",
            "TRS2", "TRB1", "TRB2", "SUBSHIFT", "KSUBTHRES", "TKSUBTHRES1", "TKSUBTHRES2", "BV", "IBV", "NBV", "RDS", "RB",
            "N", "TT", "EG", "XTI", "IS", "VJ", "VP", "CJO", "MJ", "FC", "CGDMIN", "CGDMAX", "A", "CGS", "RTHJC", "RTHCA", "CTHJ"};
        if (!m.given.empty() || !m.osdiParams.empty())
            throw std::runtime_error(m.instanceName + ": a VDMOS takes M, TEMP, DTEMP and THERMAL only");
        if (m.thermal != (!m.threeTerm && m.soiNodes.size() == 1))
            throw std::runtime_error(m.instanceName + ": a VDMOS has 3 nodes (d g s), or 5 (d g s tj tc) with THERMAL");
        if (m.thermal && !c.params.count("RTHJC"))
            throw std::runtime_error(m.instanceName + ": THERMAL needs RTHJC on model " + c.name);
        std::map<std::string, double> p;
        for (const auto& [k, v] : c.params) {
            if (soa.count(k)) continue;
            const auto a = alias.find(k);
            const std::string key = a == alias.end() ? k : a->second;
            if (!known.count(key)) throw std::runtime_error("model " + c.name + ": parameter " + k + " is not supported yet");
            p[key] = v;
        }
        bounded(&c, {"RD", "RS", "RG", "RB", "RQ", "TT", "CJO", "CGDMIN", "CGDMAX", "CGS", "CTHJ", "IS", "IBV", "KF"},
                {"PHI", "N", "NBV", "KSUBTHRES", "RTHJC", "RTHCA"});
        if (!(c.get("M", 0.5) < 1.0 && c.get("VJ", 0.8) > 0.0))
            throw std::runtime_error("model " + c.name + ": needs M < 1, VJ > 0");
        p["TYPE"] = c.type == "VDMOSP" ? -1.0 : 1.0;
        if (!p.count("TNOM")) p["TNOM"] = tnomC_;
        if (!osdi::findModule("vdmos_ng")) osdi::loadLibrary("vdmos.osdi");
        auto& model = osdiModels_[toUpper(c.name) + " VDMOS"];
        if (!model) model = std::make_shared<osdi::Model>(osdi::findModule("vdmos_ng"), c.name, p);
        OsdiDevice o;
        o.instanceName = m.instanceName;
        o.modelName = c.name;
        o.nodes = {m.drain, m.gate, m.source, m.thermal ? m.body : 0, m.thermal ? m.soiNodes[0] : 0};
        if (m.thermal) o.params.push_back({"thermal", 1.0});
        if (m.extra.mult != 1.0) { o.params.push_back({"m", m.extra.mult}); o.params.push_back({"mult_i", m.extra.mult}); }
        // ngspice: DTEMP wins over TEMP when both are given; under self-heating TEMP sets the ambient
        if (m.inst.dtemp != 0.0) o.params.push_back({"dtemp_i", m.inst.dtemp});
        else if (!std::isnan(m.inst.temp)) o.params.push_back({"temp_i", m.inst.temp + 273.15});
        o.model = model;
        return o;
    };
    auto resolveBlock = [&](TensorBlock& blk) {
        // BSIM4 (LEVEL=54 or 14) and BSIM3 (LEVEL=49 or 8) cards, ngspice's native models, run on the OSDI
        // bsim4va / bsim3_va (OpenVAF, loaded from TREESPICE_OSDI_PATH as bsim4.osdi / bsim3.osdi when needed).
        // Their geometry parameters are model parameters there, so each card plus instance geometry is one
        // OSDI model (shared by equal devices).
        // family: 3 BSIM3, 4 BSIM4, 10 BSIM-SOI (bsimsoi, B4SOI LEVEL 10/58: terminals d g s e [p [b [t]]])
        auto bsimOsdi = [&](const Mosfet& m, const ModelCard& c, int family) {
            const bool v3 = family == 3;
            const char* module = v3 ? "bsim3_va" : family == 4 ? "bsim4va" : "bsimsoi";
            const std::string tag = v3 ? "BSIM3" : family == 4 ? "BSIM4" : "BSIM-SOI";
            if (!std::isnan(m.inst.temp) || m.inst.dtemp != 0.0)
                throw std::runtime_error(m.instanceName + ": TEMP/DTEMP on a " + tag + " device is not supported yet");
            if (family != 10 && !m.soiNodes.empty())
                throw std::runtime_error(m.instanceName + ": expected 4 nodes (d g s b) for a " + tag + " model");
            if (!osdi::findModule(module)) osdi::loadLibrary(v3 ? "bsim3.osdi" : family == 4 ? "bsim4.osdi" : "bsimsoi.osdi");
            std::map<std::string, double> p;
            for (const auto& [k, v] : c.params)  // binning bounds select the card; LEVEL names it
                if (k != "LEVEL" && k != "LMIN" && k != "LMAX" && k != "WMIN" && k != "WMAX" && k != "LINTNOI") p[k] = v;
            // ponytail: LINTNOI (a BSIM4.8 flicker-noise length offset) is unknown to
            // bsim4va and dropped; only flicker noise is affected.
            p["TYPE"] = c.type == "PMOS" ? -1.0 : 1.0;
            if (!p.count("TNOM")) p["TNOM"] = tnomC_;  // ngspice: the model's TNOM defaults to .options tnom
            if (v3) {
                for (const char* g : {"AGIDL", "LAGIDL", "WAGIDL", "PAGIDL"})  // bsim3_va's GIDL differs from b3ld.c (3.5e-8 A vs 9e-11 A at Vd 3 V, Vg 0)
                    if (p.count(g) && p[g] != 0.0) throw std::runtime_error("model " + c.name + ": parameter " + g + " is not supported yet (BSIM3)");
                if (p.count("NQSMOD") && p["NQSMOD"] != 0.0)  // bsim3_va's charge-deficit NQS differs from b3ld.c in AC (0.4 vs 0.3 imaginary at 1 GHz)
                    throw std::runtime_error("model " + c.name + ": NQSMOD is not supported yet (BSIM3)");
                if (p.count("VERSION") && p["VERSION"] < 3.2) throw std::runtime_error("model " + c.name + ": BSIM3 VERSION below 3.2 is not supported yet");
                if (p.count("NPEAK")) { p["NCH"] = p["NPEAK"]; p.erase("NPEAK"); }  // ngspice's name for the peak channel doping
                // b3temp.c with ACM = 0: S/D resistance is RSH * squares (NRD / NRS default 1); RD, RS, RDC, RSC are
                // ACM > 0 inputs. bsim3_va's ACM = 0 uses RD + RDC and ignores RSH, so hand it the product.
                if (!p.count("ACM") || p["ACM"] == 0.0) {
                    auto sq = [&](const char* k) { const auto it = m.given.find(k); return it != m.given.end() ? it->second : 1.0; };
                    const double rsh = p.count("RSH") ? p["RSH"] : 0.0;
                    p["NRD"] = sq("NRD"); p["NRS"] = sq("NRS");  // the VA's S/D thermal noise needs squares > 0
                    p["RD"] = rsh * p["NRD"]; p["RS"] = rsh * p["NRS"]; p["RDC"] = 0.0; p["RSC"] = 0.0;
                }
            }
            std::string key = toUpper(c.name);
            auto given = m.given;  // geometry, then what only the OSDI model knows (BSIM-SOI SOIMOD, RTH0, ...)
            given.insert(m.osdiParams.begin(), m.osdiParams.end());
            for (const auto& [k, v] : given) {
                if (k == "NF" && v == 1.0) continue;
                p[k] = v;
                std::ostringstream kv;
                kv << " " << k << "=" << std::setprecision(17) << v;
                key += kv.str();
            }
            // ponytail: ngspice runs VERSION < 4.6 cards on BSIM4v5, whose junction saturation current is 1e-14 A
            // when the effective area and perimeter are both <= 0; bsim4va (4.8) leaves it 0, so a forward-biased
            // body draws nothing. An area giving 1e-14 A at this temperature (b4temp TEMPMOD 0) stands in for
            // instances that give AS/PS (AD/PD) as 0, as sky130's subcircuits do, when the gate-sidewall JSWGS is 0.
            // Geometry left to BSIM4's defaults is not covered (its 4.5 and 4.8 defaults differ, BUG_807).
            if (family == 4 && p.count("VERSION") && p["VERSION"] < 4.6 && (!p.count("TEMPMOD") || p["TEMPMOD"] == 0.0)) {
                auto get = [&](const char* k, double d) { return p.count(k) ? p[k] : d; };
                const double Tn = p["TNOM"] + 273.15, T = ambientK_;
                auto eg = [&](double t) { return get("BG0SUB", 1.16) - get("TBGASUB", 7.02e-4) * t * t / (t + get("TBGBSUB", 1108.0)); };
                auto zeroGeo = [&](const char* a, const char* pr) {
                    const auto ia = m.given.find(a), ip = m.given.find(pr);
                    return ia != m.given.end() && ip != m.given.end() && ia->second <= 0.0 && ip->second <= 0.0;
                };
                auto shim = [&](const char* area, const char* per, double js, double xti, double nj, double jswg) {
                    if (!zeroGeo(area, per) || js <= 0.0 || jswg != 0.0) return false;
                    const double jsT = js * std::exp((eg(Tn) / devtemp::vt(Tn) - eg(T) / devtemp::vt(T) + xti * std::log(T / Tn)) / nj);
                    p[area] = 1e-14 / jsT;
                    return true;
                };
                const double jss = get("JSS", 1e-4), xtis = get("XTIS", 3.0), njs = get("NJS", 1.0), cjs = get("CJS", 5e-4);
                const bool s = shim("AS", "PS", jss, xtis, njs, get("JSWGS", 0.0));
                const bool d = shim("AD", "PD", get("JSD", jss), get("XTID", xtis), get("NJD", njs), get("JSWGD", get("JSWGS", 0.0)));
                // The stand-in area must not add bottom-junction capacitance (CJS/CJD * area): that side had none
                if (s) { if (!d && !p.count("CJD")) p["CJD"] = cjs; p["CJS"] = 0.0; }
                if (d) p["CJD"] = 0.0;
            }
            auto& model = osdiModels_[key];
            if (!model) model = std::make_shared<osdi::Model>(osdi::findModule(module), c.name, p);
            OsdiDevice o;
            o.instanceName = m.instanceName;
            o.modelName = c.name;
            o.nodes = {m.drain, m.gate, m.source, m.body};
            o.nodes.insert(o.nodes.end(), m.soiNodes.begin(), m.soiNodes.end());
            if (m.extra.mult != 1.0) o.params.push_back({"m", m.extra.mult});
            o.model = model;
            return o;
        };
        // first(c, {"A", "B"}, def): the first alias the card gives (SPICE spellings vary)
        auto first = [](const ModelCard* c, std::initializer_list<const char*> keys, double def) {
            for (const char* k : keys) if (c->params.count(k)) return c->params.at(k);
            return def;
        };
        for (auto& s : blk.switches) {
            const ModelCard* c = card(s.modelName);
            const bool w = !s.ctrlName.empty();
            if (c->type != (w ? "CSW" : "SW"))
                throw std::runtime_error(s.name + ": model " + c->name + " is " + c->type + ", not a " + (w ? "CSW" : "SW") + " model");
            if (w) supported(c, {"IT", "IH", "RON", "ROFF"}); else supported(c, {"VT", "VH", "RON", "ROFF"});
            s.vt = c->get(w ? "IT" : "VT", 0.0); s.vh = c->get(w ? "IH" : "VH", 0.0);
            s.ron = c->get("RON", 1.0); s.roff = c->get("ROFF", 1e12);
            if (!(s.ron > 0.0 && s.roff > 0.0)) throw std::runtime_error("model " + c->name + ": needs RON, ROFF > 0");
        }
        // Resistors (ngspice restemp): a model card gives the value from its sheet resistance and the
        // geometry, R = RSH (L - 2 SHORT) / (W - 2 NARROW), unless the instance gives it; TC1/TC2 (the
        // instance's, else the model's) scale it from the model's TNOM to the instance temperature.
        for (auto& r : blk.resistors) {
            double tnom = tnomK, tc1 = r.tc1, tc2 = r.tc2;
            if (!r.model_name.empty()) {
                const ModelCard* c = card(r.model_name);
                if (c->type != "R") throw std::runtime_error(r.name + ": model " + c->name + " is " + c->type + ", not an R model");
                supported(c, {"RSH", "TC1", "TC2", "TNOM", "DEFW", "W", "L", "NARROW", "SHORT"});
                if (std::isnan(r.resistance_ohms)) {
                    const double w = (r.width_m > 0 ? r.width_m : first(c, {"W", "DEFW"}, 10e-6)) - 2.0 * c->get("NARROW", 0.0);
                    const double l = (r.length_m > 0 ? r.length_m : c->get("L", 10e-6)) - 2.0 * c->get("SHORT", 0.0);
                    r.resistance_ohms = c->get("RSH", 0.0) * l / w;
                    if (!(r.resistance_ohms != 0.0 && std::isfinite(r.resistance_ohms)))
                        throw std::runtime_error(r.name + ": model " + c->name + " gives a resistance of " +
                                                 std::to_string(r.resistance_ohms) + " (needs RSH, L and W)");
                }
                tnom = c->get("TNOM", tnomC_) + 273.15;
                if (!r.tc1Given) tc1 = c->get("TC1", 0.0);
                if (!r.tc2Given) tc2 = c->get("TC2", 0.0);
                r.model_name.clear();
            } else if (std::isnan(r.resistance_ohms)) throw std::runtime_error(r.name + ": no value and no model");
            r.resistance_ohms /= r.m;
            r.ac_ohms /= r.m;
            r.m = 1.0;
            r.tcFactor = devtemp::poly(tc1, tc2, instK(r.inst) - tnom);
            r.resistance_ohms *= r.tcFactor;
        }
        // Capacitors: a model card gives the value (ngspice capsetup/captemp) unless the
        // instance does; M multiplies. TC1/TC2 (the instance's, else the model's) scale it from TNOM.
        for (auto& cap : blk.capacitors) {
            double tnom = tnomK, tc1 = cap.tc1, tc2 = cap.tc2;
            if (!cap.model_name.empty()) {
                const ModelCard* c = card(cap.model_name);
                if (c->type != "C") throw std::runtime_error(cap.name + ": model " + c->name + " is " + c->type + ", not a C model");
                supported(c, {"CAP", "CJ", "CJSW", "DEFW", "DEFL", "NARROW", "SHORT", "TNOM", "TC1", "TC2"});
                const double w = (cap.width_m > 0 ? cap.width_m : c->get("DEFW", 1e-6)) - c->get("NARROW", 0.0);
                const double l = (cap.length_m > 0 ? cap.length_m : c->get("DEFL", 0.0)) - c->get("SHORT", 0.0);
                if (std::isnan(cap.capacitance_farads))
                    cap.capacitance_farads = c->params.count("CAP") ? c->get("CAP", 0.0)
                                                                    : c->get("CJ", 0.0) * w * l + 2.0 * c->get("CJSW", 0.0) * (l + w);
                tnom = c->get("TNOM", tnomC_) + 273.15;
                if (!cap.tc1Given) tc1 = c->get("TC1", 0.0);
                if (!cap.tc2Given) tc2 = c->get("TC2", 0.0);
                cap.model_name.clear();
            } else if (std::isnan(cap.capacitance_farads)) throw std::runtime_error(cap.name + ": no value and no model");
            cap.capacitance_farads *= devtemp::poly(tc1, tc2, instK(cap.inst) - tnom);
            cap.capacitance_farads *= cap.m * cap.scale;
            cap.m = 1.0;
        }
        std::vector<Mosfet> level1;
        for (auto& m : blk.mosfets) {
            const bool icGiven = !std::isnan(m.ic[0]) || !std::isnan(m.ic[1]) || !std::isnan(m.ic[2]);
            auto native = [&](const char* what) {
                if (m.off || icGiven) throw std::runtime_error(m.instanceName + ": OFF and IC on a " + what + " device are not supported yet");
            };
            if (const ModelCard* v = mosCard(m); v->type == "VDMOSN" || v->type == "VDMOSP") {
                native("VDMOS");
                sensList(blk, m.instanceName, *v);
                blk.osdiDevices.push_back(vdmosOsdi(m, *sensed(v, m.instanceName)));
                continue;
            }
            if (m.threeTerm || m.thermal)
                throw std::runtime_error(m.instanceName + ": expected 4 nodes (d g s b); 3 nodes and THERMAL are for a VDMOS model");
            if (sens_ && !sens_->model && toUpper(m.instanceName) == sens_->instance)
                (sens_->param == "W" ? m.w : m.l) = m.given[sens_->param] = sens_->value;
            const ModelCard* c = kind(mosCard(m), m.instanceName, {"NMOS", "PMOS"});
            sensList(blk, m.instanceName, *c, &m.given);
            if (const double level = c->get("LEVEL", 1.0); level == 54.0 || level == 14.0 || level == 49.0 || level == 8.0 ||
                                                           level == 10.0 || level == 58.0) {
                native(("LEVEL " + std::to_string((int)level)).c_str());
                blk.osdiDevices.push_back(bsimOsdi(m, *c, level == 49.0 || level == 8.0 ? 3 : level == 10.0 || level == 58.0 ? 10 : 4));
                continue;
            }
            // LEVEL 9 is ngspice's mos9: mos3 with its effective L and W required above 1 um (mos9temp)
            // ponytail: mos9load also drops mos3load's argb == 0 guard on dldvd / dldem (a degenerate bias); level 3's applies
            const int levelIn = (int)c->get("LEVEL", 1.0), level = levelIn == 9 ? 3 : levelIn;
            if (level >= 55 && level <= 57)
                throw std::runtime_error("model " + c->name + ": LEVEL " + std::to_string(level) + " (B3SOI) is not supported yet; B4SOI is LEVEL 10/58");
            if (!m.soiNodes.empty()) throw std::runtime_error(m.instanceName + ": expected 4 nodes (d g s b) for a LEVEL " + std::to_string(level) + " model");
            if (!m.osdiParams.empty()) throw std::runtime_error(m.instanceName + ": parameter " + m.osdiParams.begin()->first + " is not supported yet");
            const bool hi = level == 2 || level == 3;  // native levels 2 and 3 (mos2*.c, mos3*.c)
            if (level != 1 && level != 6 && !hi) throw std::runtime_error("model " + c->name + ": LEVEL " + std::to_string(level) + " is not supported yet");
            if (m.extra.nf != 1) throw std::runtime_error(m.instanceName + ": NF is not supported yet (level " + std::to_string(level) + ")");
            // RD, RS and RSH became series resistors in mosSeries(); JSSW is no MOS1 parameter in ngspice
            // (it warns and ignores it), so it stays an error rather than a silently dropped one.
            if (hi) {  // ponytail: KF / AF (level 2/3 flicker uses Leff^2, not W L), DELVTO, UTRA stay explicit errors
                std::vector<const char*> used = {"LEVEL", "KP", "VTO", "GAMMA", "PHI", "IS", "LD", "TOX", "UO", "U0", "CGSO", "CGDO",
                              "CGBO", "CBD", "CBS", "CJ", "MJ", "CJSW", "MJSW", "PB", "FC", "RD", "RS", "RSH", "JS", "NSUB", "NSS",
                              "TPG", "NFS", "DELTA", "XJ", "VMAX", "TNOM"};
                for (const char* x : level == 2 ? std::vector<const char*>{"LAMBDA", "UCRIT", "UEXP", "NEFF"}
                                                : std::vector<const char*>{"ETA", "THETA", "KAPPA", "XL", "WD", "XW"}) used.push_back(x);
                supported(c, used);
            } else if (level == 6)  // mos6*.c; ngspice reads NVTH and PS but no equation uses them (mos6load)
                supported(c, {"LEVEL", "VTO", "VT0", "KV", "NV", "KC", "NC", "NVTH", "PS", "GAMMA", "GAMMA1", "SIGMA", "PHI",
                              "LAMBDA", "LAMBDA0", "LAMBDA1", "RD", "RS", "CBD", "CBS", "IS", "PB", "CGSO", "CGDO", "CGBO",
                              "RSH", "CJ", "MJ", "CJSW", "MJSW", "JS", "LD", "TOX", "U0", "UO", "FC", "TPG", "NSUB", "NSS", "TNOM"});
            else
            supported(c, {"KP", "VTO", "LAMBDA", "GAMMA", "PHI", "IS", "LD", "TOX", "UO", "U0",
                          "CGSO", "CGDO", "CGBO", "CBD", "CBS", "CJ", "MJ", "CJSW", "MJSW", "PB", "FC",
                          "RD", "RS", "RSH", "JS", "NSUB", "KF", "AF", "TNOM"});
            bounded(c, {"KP", "LAMBDA", "GAMMA", "IS", "LD", "TOX", "UO", "U0", "CGSO", "CGDO", "CGBO",
                        "CBD", "CBS", "CJ", "CJSW", "MJ", "MJSW", "RD", "RS", "RSH", "JS", "NSUB", "KF",
                        "NFS", "NSS", "DELTA", "XJ", "UCRIT", "UEXP", "VMAX", "ETA", "THETA", "KAPPA"},
                    {"PHI", "NEFF"});
            if (hi && c->params.count("TOX") && !(c->params.at("TOX") > 0.0)) throw std::runtime_error("model " + c->name + ": TOX must be > 0");
            auto& k = m.card;
            k.resolved = true;
            k.pmos = c->type == "PMOS";
            k.level = hi || level == 6 ? level : 1;
            k.tox = c->get("TOX", hi ? 1e-7 : 0.0);  // mos2set / mos3set: 100 nm unless given
            // No KP with TOX: KP = UO[cm^2/Vs] * COX (ngspice mos1temp)
            k.kp = c->get("KP", k.tox > 0.0 ? c->get("UO", c->get("U0", 600.0)) * 1e-4 * 3.9 * 8.854214871e-12 / k.tox
                                            : 2e-5);
            if (level == 6) {  // mos6set / mos6temp: KC (in kp) from U0 and TOX unless given, LAMBDA0 over LAMBDA
                k.kp = c->get("KC", k.tox > 0.0 ? 0.5 * c->get("UO", c->get("U0", 600.0)) * 1e-4 * 3.9 * 8.854214871e-12 / k.tox : 5e-5);
                k.kv = c->get("KV", 2.0); k.nv = c->get("NV", 0.5); k.nc = c->get("NC", 1.0);
                k.gamma1 = c->get("GAMMA1", 0.0); k.sigma = c->get("SIGMA", 0.0); k.lambda1 = c->get("LAMBDA1", 0.0);
            }
            k.vto = first(c, {"VTO", "VT0"}, 0.0); k.lambda = first(c, {"LAMBDA0", "LAMBDA"}, 0.0); k.gamma = c->get("GAMMA", 0.0);
            k.phi = c->get("PHI", 0.6); k.is = c->get("IS", 1e-14); k.ld = c->get("LD", 0.0);
            k.js = c->get("JS", 0.0); k.kf = c->get("KF", 0.0); k.af = c->get("AF", 1.0);
            if (k.kf != 0.0 && !(k.tox > 0.0)) throw std::runtime_error("model " + c->name + ": KF needs TOX (flicker noise divides by the oxide capacitance)");
            const double Tn = c->get("TNOM", tnomC_) + 273.15, T = instK(m.inst);
            if (hi) {
                k.mu = c->get("UO", c->get("U0", 600.0)) * 1e-4;
                k.nfs = c->get("NFS", 0.0); k.delta = c->get("DELTA", 0.0); k.xj = c->get("XJ", 0.0); k.vmax = c->get("VMAX", 0.0);
                k.ucrit = c->get("UCRIT", 1e4); k.uexp = c->get("UEXP", 0.0); k.neff = c->get("NEFF", 1.0);
                k.eta = c->get("ETA", 0.0); k.theta = c->get("THETA", 0.0); k.kappa = c->get("KAPPA", 0.2);
                k.xl = c->get("XL", 0.0); k.wd = c->get("WD", 0.0); k.xw = c->get("XW", 0.0);
            }
            if (c->params.count("NSUB")) {
                // ngspice mos1temp at TNOM = 27 C: PHI, GAMMA and VTO from the substrate doping unless given
                // (aluminium-free n+ gate, NSS = 0). Levels 2 and 3 add TPG and NSS.
                const double nsub = c->get("NSUB", 0.0) * 1e6, vt = devtemp::vt(Tn), type = k.pmos ? -1.0 : 1.0;
                const double eg = devtemp::egfet(Tn), ni = 1.45e16;
                if (!(k.tox > 0.0) || !(nsub > ni))
                    throw std::runtime_error("model " + c->name + ": NSUB needs TOX and NSUB > 1.45e10 cm^-3 (ni)");
                const double cox = 3.9 * 8.854214871e-12 / k.tox;
                if (!c->params.count("PHI")) k.phi = std::max(0.1, 2.0 * vt * std::log(nsub / ni));
                if (!c->params.count("GAMMA"))
                    k.gamma = std::sqrt(2.0 * 11.7 * 8.854214871e-12 * PhysicsConstants::Q_ELECTRON * nsub) / cox;
                if (!c->params.count("VTO") && !c->params.count("VT0")) {
                    const double tpg = c->get("TPG", 1.0), fermis = type * 0.5 * k.phi, fermig = type * tpg * 0.5 * eg;
                    const double wkfngs = (tpg != 0.0 ? 3.25 + 0.5 * eg - fermig : 3.2) - (3.25 + 0.5 * eg + fermis);
                    k.vto = wkfngs - c->get("NSS", 0.0) * 1e4 * PhysicsConstants::Q_ELECTRON / cox + type * (k.gamma * std::sqrt(k.phi) + k.phi);
                }
                if (hi) { k.nsub = nsub; k.xd = std::sqrt(2.0 * 11.7 * 8.854214871e-12 / (PhysicsConstants::Q_ELECTRON * nsub)); k.alpha = k.xd * k.xd; }
            }
            k.cgso = c->get("CGSO", 0.0); k.cgdo = c->get("CGDO", 0.0); k.cgbo = c->get("CGBO", 0.0);
            k.cbd = c->get("CBD", 0.0); k.cbs = c->get("CBS", 0.0); k.cj = c->get("CJ", 0.0);
            k.cjsw = c->get("CJSW", 0.0); k.mj = c->get("MJ", 0.5); k.mjsw = c->get("MJSW", hi ? 0.33 : 0.5);  // mos2set / mos3set grade the sidewall 0.33, mos1set 0.5
            k.pb = c->get("PB", 0.8); k.fc = c->get("FC", 0.5);
            const double lmin = levelIn == 9 ? 1e-6 : 0.0;
            if (!(m.l - 2.0 * k.ld + k.xl > lmin && m.w - 2.0 * k.wd + k.xw > lmin))
                throw std::runtime_error(m.instanceName + ": effective L (L - 2*LD + XL) and W (W - 2*WD + XW) must be > " +
                                         (levelIn == 9 ? std::string("1 um (LEVEL 9)") : std::string("0")));
            if (!(k.mj < 1.0 && k.mjsw < 1.0 && k.pb > 0.0 && k.fc < 1.0))
                throw std::runtime_error("model " + c->name + ": needs MJ, MJSW < 1, PB > 0, FC < 1");
            // mos1temp: KP/T^1.5, PHI, VTO (via the flat-band voltage), IS and the junctions at the instance temperature
            {
                using namespace devtemp;
                const double ratio = T / Tn, eg = egfet(T), eg1 = egfet(Tn), sgn = k.pmos ? -1.0 : 1.0;
                k.vbi = k.vto - sgn * k.gamma * std::sqrt(k.phi);
                if (T != Tn) {
                    const double phio = (k.phi - pbfact(Tn)) / (Tn / REFTEMP), tphi = T / REFTEMP * phio + pbfact(T);
                    k.vbi += 0.5 * (eg1 - eg) + sgn * 0.5 * (tphi - k.phi);
                    k.vto = k.vbi + sgn * k.gamma * std::sqrt(tphi);
                    k.phi = tphi;
                    k.kp /= ratio * std::sqrt(ratio);
                    if (level == 3) k.mu /= ratio * std::sqrt(ratio);
                    k.is *= std::exp(-eg / vt(T) + eg1 / vt(Tn));
                    k.js *= std::exp(-eg / vt(T) + eg1 / vt(Tn));
                    double pbSide = k.pb;  // the same tBulkPot as k.pb gets; only the sidewall's grading differs
                    const double side = junctionAt(pbSide, k.mjsw, Tn, T);
                    const double bot = junctionAt(k.pb, k.mj, Tn, T);
                    k.cbd *= bot; k.cbs *= bot; k.cj *= bot; k.cjsw *= side;
                }
            }
            k.vt = devtemp::vt(T);
            level1.push_back(m);
        }
        blk.mosfets = std::move(level1);
        // Diode and BJT at the instance temperature (ngspice diotemp / bjttemp, TLEV = TLEVC = 0): IS with EG and XTI,
        // junction potentials and capacitances, betas with XTB, the TC-style coefficients; a card at TNOM stays as given.
        for (auto& d : blk.diodes) if (const ModelCard* c = kind(card(d.modelName), d.instanceName, {"D", "D"})) {
            sensList(blk, d.instanceName, *c);
            supported(c, {"IS", "JS", "N", "RS", "CJO", "CJ0", "CJ", "VJ", "PB", "M", "MJ", "FC", "TT", "BV", "VB", "VRB", "VAR",
                          "IBV", "IB", "EG", "XTI", "TNOM", "TM1", "TM2", "TCV", "TBV1", "TTT1", "TTT2", "TRS", "TRS1", "TRS2",
                          "JSW", "ISW", "NS", "CJSW", "MJSW", "PHP", "VJSW", "IKF", "IK", "IKR", "IKP", "KF", "AF", "ISR", "NR",
                          "NBV", "NZ", "CTA", "CTC", "CTP", "TPB", "TVJ", "TPHP", "TLEVC", "TLEV", "CJP", "FCS", "GAP1", "GAP2",
                          "JTUN", "JTUNSW", "NTUN", "XTITUN", "KEG", "AREA", "PJ", "TREF", "VP", "RSW",
                          "LEVEL", "LM", "LP", "WM", "WP", "XOM", "XOI", "XM", "XP", "XW",
                          "FV_MAX", "BV_MAX", "ID_MAX", "TE_MAX", "PD_MAX",  // SOA: ngspice's warnings only
                          "RTH0", "CTH0"});  // self-heating: a THERMAL instance (D a k tj) uses them
            // Temperature selectors (probed on ngspice 47): TLEV 1 scales BV by (1 - TCV dT) where 0 subtracts TCV dT
            // (IS follows the same law under both); TLEV 2 also takes the band gap EG - GAP1 T^2 / (T + GAP2) for the
            // saturation currents (exp(EG(Tn)/(N Vt(Tn)) - EG(T)/(N Vt) + XTI/N ln(T/Tn))) and the junction potentials;
            // TLEVC 1 takes CJO (1 + CTA dT), VJ - TPB dT and the sidewall's CTP / TPHP in place of the junction law,
            // and CTA..TPHP do nothing under TLEVC 0
            const double tlev = c->get("TLEV", 0.0), tlevc = c->get("TLEVC", 0.0);
            if ((tlev != 0.0 && tlev != 1.0 && tlev != 2.0) || (tlevc != 0.0 && tlevc != 1.0))
                throw std::runtime_error("model " + c->name + ": TLEV other than 0-2 / TLEVC other than 0-1 is not supported yet");
            bounded(c, {"IS", "JS", "RS", "RSW", "CJO", "CJ0", "CJ", "M", "MJ", "TT", "JSW", "ISW", "CJSW", "MJSW", "IKF", "IK", "IKR",
                        "IKP", "KF", "ISR", "JTUN", "JTUNSW", "AREA", "PJ"},
                    {"N", "BV", "VB", "VRB", "VAR", "NS", "PHP", "VJSW", "NR", "NBV", "NZ", "NTUN"});  // ngspice: BV = 0 breaks down at 0 V
            // diosetup: the model's AREA / PJ unless the instance gives one, or gives W or L (then 1 and 0)
            const double level = c->get("LEVEL", 1.0);
            if (level != 1.0 && level != 3.0) throw std::runtime_error("model " + c->name + ": diode LEVEL other than 1 or 3 is not supported");
            const bool wl = !std::isnan(d.w) || !std::isnan(d.l);
            if (!d.areaGiven && !wl) d.area *= c->get("AREA", 1.0);
            if (!d.pjGiven) d.pj = wl ? 0.0 : c->get("PJ", 0.0);
            if (level == 3.0) {
                // LEVEL 3 (ngspice diosetup): W and L give AREA = (W+XW)(L+XW) and PJ = 2(W+XW) + 2(L+XW) (.option
                // scale on the lengths), and the metal / poly overlaps eps_SiO2/XOM (WM+XM)(LM+XM) M (XOI, WP, LP, XP)
                // across the junction, XOM and XOI in Angstrom. Deliberate divergences: ngspice multiplies M into that
                // AREA and PJ and then again where they scale the device (M^2), here M scales it once (d.area holds the
                // instance's M); ngspice's XOM / XOI default 1e4 is taken in metres (no overlap capacitance), here the
                // 1e4 Angstrom it documents
                d.card.level3 = true;
                const double xw = c->get("XW", 0.0), s = scale_;
                if (!std::isnan(d.w) && !std::isnan(d.l)) {
                    d.area = (d.w + xw) * (d.l + xw) * s * s * d.mult;  // in place of a given AREA
                    d.pj = (2.0 * (d.w + xw) + 2.0 * (d.l + xw)) * s;
                }
                const double epsOx = 3.9 / (4e-7 * M_PI * 299792458.0 * 299792458.0);  // ngspice CONSTepsSiO2
                auto ov = [&](double inst, const char* key) { return std::isnan(inst) ? c->get(key, 0.0) : inst; };
                const double xm = c->get("XM", 0.0), xp = c->get("XP", 0.0);
                d.card.cpar = epsOx / (c->get("XOM", 1e4) * 1e-10) * d.mult * (ov(d.wm, "WM") * s + xm) * (ov(d.lm, "LM") * s + xm) +
                              epsOx / (c->get("XOI", 1e4) * 1e-10) * d.mult * (ov(d.wp, "WP") * s + xp) * (ov(d.lp, "LP") * s + xp);
            } else if (!std::isnan(d.lm) || !std::isnan(d.lp) || !std::isnan(d.wm) || !std::isnan(d.wp))
                throw std::runtime_error(d.instanceName + ": LM, LP, WM, WP are LEVEL 3 parameters");
            d.saturationCurrent_I_S_A = first(c, {"IS", "JS"}, 1e-14);
            d.emissionCoefficient_N   = c->get("N", 1.0);
            auto& k = d.card;
            k.rs = c->get("RS", 0.0); k.cjo = first(c, {"CJO", "CJ0", "CJ"}, 0.0);
            k.vj = first(c, {"VJ", "PB"}, 1.0); k.m = first(c, {"M", "MJ"}, 0.5);
            k.fc = c->get("FC", 0.5); k.tt = c->get("TT", 0.0);
            k.jsw = first(c, {"JSW", "ISW"}, 0.0); k.cjsw = first(c, {"CJSW", "CJP"}, 0.0);
            k.nsGiven = c->params.count("NS") > 0; k.ns = c->get("NS", 1.0);  // NS's default 1 sets the sidewall's IS law
            k.mjsw = c->get("MJSW", 0.33); k.php = first(c, {"PHP", "VJSW"}, 1.0); k.fcs = c->get("FCS", 0.5);
            k.ikf = first(c, {"IKF", "IK"}, 0.0); k.ikr = c->get("IKR", 0.0); k.ikp = c->get("IKP", 0.0);
            k.kf = c->get("KF", 0.0); k.af = c->get("AF", 1.0);
            k.bvGiven = c->params.count("BV") || c->params.count("VB") || c->params.count("VRB") || c->params.count("VAR");
            k.bv = first(c, {"BV", "VB", "VRB", "VAR"}, 0.0); k.ibv = first(c, {"IBV", "IB"}, 1e-3);
            k.isr = c->get("ISR", 0.0); k.nr = c->get("NR", 2.0); k.nbv = first(c, {"NBV", "NZ"}, NAN);
            k.jtun = c->get("JTUN", 0.0); k.jtunsw = c->get("JTUNSW", 0.0); k.ntun = c->get("NTUN", 30.0);
            k.vp = c->get("VP", 0.0);
            k.rswGiven = c->params.count("RSW") > 0; k.rsw = c->get("RSW", 0.0);
            // ngspice lumps a separate sidewall's charge onto the bottom junction under soft recovery (dioload), whose
            // small-signal capacitance then is not that charge's
            if (k.rswGiven && k.qp()) throw std::runtime_error("model " + c->name + ": VP with RSW is not supported yet");
            if (!(k.m < 1.0 && k.mjsw < 1.0 && k.vj > 0.0 && k.fc < 1.0 && k.fcs < 1.0 && k.rs >= 0.0 && k.ibv > 0.0))
                throw std::runtime_error("model " + c->name + ": needs M, MJSW < 1, VJ > 0, FC, FCS < 1, RS >= 0, IBV > 0");
            devtemp::DiodeLaw l;
            l.tn = first(c, {"TNOM", "TREF"}, tnomC_) + 273.15;
            l.eg = std::max(c->get("EG", 1.11), 0.1); l.xti = c->get("XTI", 3.0);
            l.keg = c->get("KEG", 1.0); l.xtitun = c->get("XTITUN", 3.0);
            l.tlev = (int)tlev; l.tlevc = (int)tlevc;
            if (tlev == 2.0) l.gap = devtemp::Gap{l.eg, c->get("GAP1", 7.02e-4), c->get("GAP2", 1108.0)};
            l.tm1 = c->get("TM1", 0.0); l.tm2 = c->get("TM2", 0.0); l.ttt1 = c->get("TTT1", 0.0); l.ttt2 = c->get("TTT2", 0.0);
            l.trs1 = first(c, {"TRS1", "TRS"}, 0.0); l.trs2 = c->get("TRS2", 0.0); l.tcv = first(c, {"TCV", "TBV1"}, 0.0);
            l.cta = first(c, {"CTA", "CTC"}, 0.0); l.tpb = first(c, {"TPB", "TVJ"}, 0.0); l.ctp = c->get("CTP", 0.0);
            l.tphp = c->get("TPHP", 0.0);
            l.n = d.emissionCoefficient_N; l.ns = k.ns; l.nr = k.nr; l.ntun = k.ntun; l.mjsw = k.mjsw; l.fc = k.fc;
            l.bvGiven = k.bvGiven;
            l.is = d.saturationCurrent_I_S_A; l.m = k.m; l.cjo = k.cjo; l.vj = k.vj; l.cjsw = k.cjsw; l.php = k.php;
            l.jsw = k.jsw; l.isr = k.isr; l.jtun = k.jtun; l.jtunsw = k.jtunsw; l.tt = k.tt; l.rs = k.rs; l.rsw = k.rsw; l.bv = k.bv;
            const double T = instK(d.inst);
            const auto a = devtemp::diodeAt(l, T);
            d.thermalVoltage_V_T_V = a.vt; d.saturationCurrent_I_S_A = a.is;
            k.m = a.m; k.cjo = a.cjo; k.vj = a.vj; k.cjsw = a.cjsw; k.php = a.php; k.jsw = a.jsw; k.isr = a.isr;
            k.jtun = a.jtun; k.jtunsw = a.jtunsw; k.tt = a.tt; k.rs = a.rs; k.rsw = a.rsw; k.bv = a.bv;
            // self-heating (ngspice: a thermal node, THERMAL and the model's RTH0; else tj is left to its circuit)
            if (d.thermal > 0 && d.thermalFlag && c->params.count("RTH0")) {
                const double rth0 = c->get("RTH0", 0.0), cth0 = c->get("CTH0", 1e-5);
                if (!(rth0 > 0.0) || cth0 < 0.0) throw std::runtime_error("model " + c->name + ": self-heating needs RTH0 > 0, CTH0 >= 0");
                d.heat = std::make_shared<const Diode::Heat>(Diode::Heat{l, T, rth0, cth0});
            }
        }
        // HICUM/L2 (LEVEL 8): OpenVAF's hicuml2.va 2.4.0, ngspice's native version (they agree to 4e-5 in ngspice),
        // through OSDI (hicum2.osdi from TREESPICE_OSDI_PATH); its parameter names are ngspice's.
        // ponytail: the VA has no junction or temperature limiting (native HICUM's pnjlim / temperature limiter), so
        // some cards that native ngspice solves fail to converge here, as in ngspice running the same OSDI model.
        {
            std::vector<BJT> gp;
            for (auto& q : blk.bjts) {
                const ModelCard* c = kind(card(q.modelName), q.instanceName, {"NPN", "PNP"});
                const double level = c->get("LEVEL", 1.0);
                if (level == 4.0 || level == 9.0) {  // VBIC: ngspice's own (tests/models/vbic_gen.py) through OSDI
                    sensList(blk, q.instanceName, *c);
                    if (q.off) throw std::runtime_error(q.instanceName + ": OFF on a VBIC device is not supported yet");
                    if (!osdi::findModule("vbic_ng")) osdi::loadLibrary("vbic.osdi");
                    static const std::map<std::string, std::string> alias = {{"TREF", "TNOM"}, {"V0", "VO"}, {"CBE0", "CBEO"},
                        {"CBC0", "CBCO"}, {"QC0", "QCO"}, {"XV0", "XVO"}, {"XRC", "XRCX"}, {"XRB", "XRBX"}};
                    static const std::set<std::string> soa = {"VBE_MAX", "BVBE", "VBC_MAX", "BVBC", "VCE_MAX", "BVCE",
                                                              "VSUB_MAX", "BVSUB", "VBCFWD", "VBEFWD", "VSUBFWD"};
                    std::map<std::string, double> p;
                    for (const auto& [k, v] : c->params) {  // the SOA limits only drive ngspice's warnings
                        if (k == "LEVEL" || soa.count(k)) continue;
                        const auto a = alias.find(k);
                        p[a == alias.end() ? k : a->second] = v;
                    }
                    p["TYPE"] = c->type == "PNP" ? -1.0 : 1.0;
                    auto& model = osdiModels_[toUpper(c->name) + " VBIC"];
                    if (!model) model = std::make_shared<osdi::Model>(osdi::findModule("vbic_ng"), c->name, p);
                    OsdiDevice o;
                    o.instanceName = q.instanceName;
                    o.modelName = c->name;
                    o.nodes = {q.nodeCollector, q.base, q.emitter, q.substrate, q.thermal};
                    if (q.area != 1.0) o.params.push_back({"area", q.area});  // ngspice SCALE = AREA * M on everything
                    const double dt = (std::isnan(q.inst.temp) ? 0.0 : q.inst.temp + 273.15 - ambientK_) + q.inst.dtemp;
                    if (dt != 0.0) o.params.push_back({"dtemp_i", dt});
                    o.model = model;
                    blk.osdiDevices.push_back(o);
                    continue;
                }
                if (level != 8.0) { gp.push_back(q); continue; }
                sensList(blk, q.instanceName, *c);
                if (!std::isnan(q.inst.temp) || q.inst.dtemp != 0.0)  // ngspice's HICUM rejects them too
                    throw std::runtime_error(q.instanceName + ": TEMP/DTEMP on a HICUM device is not supported");
                // ngspice's native AREA / M scaling is not M parallel devices (OSDI's m): with self-heating they
                // differ by 0.2 % at 19 mK and 4 % at 19 K, so the engine would not match ngspice
                if (q.area != 1.0) throw std::runtime_error(q.instanceName + ": AREA/M on a HICUM device is not supported yet");
                if (!osdi::findModule("hicumL2va")) osdi::loadLibrary("hicum2.osdi");
                std::map<std::string, double> p;
                for (const auto& [k, v] : c->params) if (k != "LEVEL") p[k] = v;
                p["TYPE"] = c->type == "PNP" ? -1.0 : 1.0;
                if (!p.count("TNOM")) p["TNOM"] = tnomC_;
                auto& model = osdiModels_[toUpper(c->name) + " HICUM"];
                if (!model) model = std::make_shared<osdi::Model>(osdi::findModule("hicumL2va"), c->name, p);
                OsdiDevice o;
                o.instanceName = q.instanceName;
                o.modelName = c->name;
                o.nodes = {q.nodeCollector, q.base, q.emitter, q.substrate, q.thermal};
                o.model = model;
                blk.osdiDevices.push_back(o);
            }
            blk.bjts = std::move(gp);
        }
        for (auto& q : blk.bjts) if (const ModelCard* c = kind(card(q.modelName), q.instanceName, {"NPN", "PNP"})) {
            sensList(blk, q.instanceName, *c);
            supported(c, {"IS", "BF", "BR", "NF", "NR", "VAF", "VA", "VAR", "VB", "IKF", "IK", "IKR",
                          "ISE", "NE", "ISC", "NC", "RB", "RC", "RE", "CJE", "VJE", "PE", "MJE", "ME",
                          "CJC", "VJC", "PC", "MJC", "MC", "XCJC", "CJS", "CCS", "VJS", "PS", "MJS", "MS",
                          "FC", "TF", "XTF", "VTF", "ITF", "TR", "EG", "XTI", "XTB", "TNOM", "TREF", "RBM", "IRB", "NK", "NKF", "KF", "AF",
                          "ISS", "NS", "SUBS", "PTF", "C2", "C4", "CSUB", "TLEV", "TLEVC", "CTE", "CTC", "CTS", "TVJE", "TVJC", "TVJS",
                          "TRB", "TRC", "TRE",
                          "TVAF1", "TVAF2", "TVAR1", "TVAR2", "TIKF1", "TIKF2", "TIKR1", "TIKR2", "TRB1", "TRB2", "TRC1", "TRC2",
                          "TRE1", "TRE2", "TRM1", "TRM2", "TIRB1", "TIRB2", "TNF1", "TNF2", "TNR1", "TNR2", "TNE1", "TNE2",
                          "TNC1", "TNC2", "TNS1", "TNS2", "TITF1", "TITF2", "TTF1", "TTF2", "TTR1", "TTR2", "TMJE1", "TMJE2",
                          "TMJC1", "TMJC2", "TMJS1", "TMJS2", "TBF1", "TBF2", "TBR1", "TBR2", "TIS1", "TIS2", "TISE1", "TISE2",
                          "TISC1", "TISC2", "TISS1", "TISS2",
                          "IBE", "IBC", "RCO", "VO", "GAMMA", "QCO", "QUASIMOD", "VG", "CN", "D",
                          "VBE_MAX", "VBC_MAX", "VCE_MAX", "IC_MAX", "IB_MAX", "PD_MAX", "TE_MAX", "RTH0"});  // SOA (RTH0 derates PD_MAX): ngspice's warnings only
            bounded(c, {"IS", "VAF", "VA", "VAR", "VB", "IKF", "IK", "IKR", "ISE", "C2", "ISC", "C4", "CJE", "CJC", "CJS", "CCS", "CSUB",
                        "MJE", "ME", "MJC", "MC", "MJS", "MS", "TF", "XTF", "VTF", "ITF", "TR", "ISS", "RBM", "IRB", "NK", "NKF", "KF"},
                    {"NE", "NC", "NS"});
            // Temperature selectors (ngspice bjttemp): TLEV 0 betas (T/Tn)^XTB, 1: 1 + XTB dT; TLEV 3 takes the saturation
            // currents to the power 1 + TIS1 dT + TIS2 dT^2 (TISE, TISC, TISS likewise). TLEVC 1: CJx (1 + CTx dT), VJx - TVJx dT.
            // ngspice 47's TLEV 2 leaves IS unset (bjttemp has no branch for it), so it stays an error.
            const double tlev = c->get("TLEV", 0.0), tlevc = c->get("TLEVC", 0.0);
            if ((tlev != 0.0 && tlev != 1.0 && tlev != 3.0) || (tlevc != 0.0 && tlevc != 1.0))
                throw std::runtime_error("model " + c->name + ": BJT TLEV other than 0, 1, 3 / TLEVC other than 0, 1 is not supported");
            q.isNPN = (c->type != "PNP");
            q.saturationCurrent_I_S_A = c->get("IS", 1e-16);
            // the IS / ISE / ISC / ISS this card gives, before the temperature laws (TLEV 3 raises them to a power)
            q.betaF = c->get("BF", 100.0);
            q.betaR = c->get("BR", 1.0);
            auto& k = q.card;
            k.nf = c->get("NF", 1.0); k.nr = c->get("NR", 1.0);
            k.vaf = first(c, {"VAF", "VA"}, 0.0); k.var = first(c, {"VAR", "VB"}, 0.0);
            k.ikf = first(c, {"IKF", "IK"}, 0.0); k.ikr = c->get("IKR", 0.0);
            k.ise = first(c, {"ISE", "C2"}, 0.0); k.ne = c->get("NE", 1.5);
            k.isc = first(c, {"ISC", "C4"}, 0.0); k.nc = c->get("NC", 2.0);
            k.rb = c->get("RB", 0.0); k.rc = c->get("RC", 0.0); k.re = c->get("RE", 0.0);
            k.cje = c->get("CJE", 0.0); k.vje = first(c, {"VJE", "PE"}, 0.75); k.mje = first(c, {"MJE", "ME"}, 0.33);
            k.cjc = c->get("CJC", 0.0); k.vjc = first(c, {"VJC", "PC"}, 0.75); k.mjc = first(c, {"MJC", "MC"}, 0.33);
            k.xcjc = c->get("XCJC", 1.0);
            k.cjs = first(c, {"CJS", "CCS", "CSUB"}, 0.0); k.vjs = first(c, {"VJS", "PS"}, 0.75); k.mjs = first(c, {"MJS", "MS"}, 0.0);
            k.fc = c->get("FC", 0.5);
            k.iss = c->get("ISS", 0.0); k.ns = c->get("NS", 1.0);
            // IBE / IBC: ngspice takes them only as a pair (one alone it ignores); here one alone is an error
            if (c->params.count("IBE") != c->params.count("IBC"))
                throw std::runtime_error("model " + c->name + ": IBE and IBC go together (ngspice ignores one alone)");
            k.ibeIbc = c->params.count("IBE") > 0;
            k.ibe = c->get("IBE", 0.0); k.ibc = c->get("IBC", 0.0);
            if (k.ibeIbc && !(k.ibe > 0.0 && k.ibc > 0.0)) throw std::runtime_error("model " + c->name + ": IBE and IBC must be > 0");
            // Kull quasi-saturation: on with RCO given (at least 0.01, as ngspice); QUASIMOD 1 adds its temperature laws
            if (c->params.count("RCO")) {
                k.rco = std::max(c->get("RCO", 0.01), 0.01);
                k.vo = c->get("VO", 10.0); k.gamma = c->get("GAMMA", 1e-11); k.qco = c->get("QCO", 0.0);
                if (!(k.vo > 0.0 && k.gamma >= 0.0 && k.qco >= 0.0)) throw std::runtime_error("model " + c->name + ": VO must be > 0, GAMMA and QCO >= 0");
                // ngspice takes RCO and QCO per device, AREA aside, times M; AREA and M are one factor here
                if (q.area != 1.0) throw std::runtime_error(q.instanceName + ": AREA / M with quasi-saturation (RCO) is not supported yet");
            }
            // SUBS (ngspice): 1 vertical (substrate junction to C), -1 lateral (to B); default by polarity
            const double subs = c->get("SUBS", q.isNPN ? 1.0 : -1.0);
            if (subs != 1.0 && subs != -1.0) throw std::runtime_error("model " + c->name + ": SUBS must be 1 or -1");
            k.lateral = subs == -1.0;
            k.tf = c->get("TF", 0.0); k.xtf = c->get("XTF", 0.0); k.vtf = c->get("VTF", 0.0);
            k.itf = c->get("ITF", 0.0); k.tr = c->get("TR", 0.0); k.ptf = c->get("PTF", 0.0);
            k.rbm = c->get("RBM", k.rb); k.irb = c->get("IRB", 0.0); k.nk = std::min(first(c, {"NKF", "NK"}, 0.5), 1.0);
            k.kf = c->get("KF", 0.0); k.af = c->get("AF", 1.0);
            k.td = k.ptf * M_PI / 180.0 * k.tf;
            const double Tn = first(c, {"TNOM", "TREF"}, tnomC_) + 273.15, T = instK(q.inst), dt = T - Tn;
            q.thermalVoltage_V_T_V = devtemp::vt(T);
            {
                using namespace devtemp;
                auto tc = [&](const char* a, const char* b) { return poly(c->get(a, 0.0), c->get(b, 0.0), dt); };
                // bjttemp: VAF, VAR, IKF, IKR, RB, RC, RE (TRB / TRC / TRE: the linear terms), RBM, IRB, the emission
                // coefficients, ITF, TF, TR and the gradings (held below 0.999) by 1 + T*1 dT + T*2 dT^2
                k.vaf *= tc("TVAF1", "TVAF2"); k.var *= tc("TVAR1", "TVAR2");
                k.ikf *= tc("TIKF1", "TIKF2"); k.ikr *= tc("TIKR1", "TIKR2");
                k.rb *= poly(first(c, {"TRB1", "TRB"}, 0.0), c->get("TRB2", 0.0), dt);
                k.rc *= poly(first(c, {"TRC1", "TRC"}, 0.0), c->get("TRC2", 0.0), dt);
                k.re *= poly(first(c, {"TRE1", "TRE"}, 0.0), c->get("TRE2", 0.0), dt);
                k.rbm *= tc("TRM1", "TRM2"); k.irb *= tc("TIRB1", "TIRB2");
                const double ne0 = k.ne, nc0 = k.nc;  // the leakage laws take the card's NE, NC
                k.nf *= tc("TNF1", "TNF2"); k.nr *= tc("TNR1", "TNR2"); k.ne *= tc("TNE1", "TNE2"); k.nc *= tc("TNC1", "TNC2");
                k.ns *= tc("TNS1", "TNS2");
                k.itf *= tc("TITF1", "TITF2"); k.tf *= tc("TTF1", "TTF2"); k.tr *= tc("TTR1", "TTR2");
                k.mje = std::min(k.mje * tc("TMJE1", "TMJE2"), 0.999);
                k.mjc = std::min(k.mjc * tc("TMJC1", "TMJC2"), 0.999);
                k.mjs = std::min(k.mjs * tc("TMJS1", "TMJS2"), 0.999);
                const double ratlog = std::log(T / Tn), factlog = (T / Tn - 1) * c->get("EG", 1.11) / vt(T) + c->get("XTI", 3.0) * ratlog;
                const double xtb = c->get("XTB", 0.0), bfac = tlev == 0.0 ? std::exp(ratlog * xtb) : tlev == 1.0 ? 1.0 + xtb * dt : 1.0;
                if (tlev == 3.0) {
                    q.saturationCurrent_I_S_A = std::pow(q.saturationCurrent_I_S_A, tc("TIS1", "TIS2"));
                    k.ibe = std::pow(k.ibe, tc("TIS1", "TIS2")); k.ibc = std::pow(k.ibc, tc("TIS1", "TIS2"));
                    k.iss = std::pow(k.iss, tc("TISS1", "TISS2"));
                    k.ise = std::pow(k.ise, tc("TISE1", "TISE2")); k.isc = std::pow(k.isc, tc("TISC1", "TISC2"));
                } else {
                    // IBE / IBC by their own emission coefficients (the card's NF, NR); ISS then takes IBC's factor
                    const double fac = std::exp(factlog), fbc = std::exp(factlog / c->get("NR", 1.0));
                    q.saturationCurrent_I_S_A *= fac; k.iss *= k.ibeIbc ? fbc : fac;
                    k.ibe *= std::exp(factlog / c->get("NF", 1.0)); k.ibc *= fbc;
                    k.ise *= std::exp(factlog / ne0) / bfac; k.isc *= std::exp(factlog / nc0) / bfac;
                }
                if (k.rco > 0.0 && c->get("QUASIMOD", 0.0) == 1.0) {  // bjttemp: RCO, VO by powers of T / Tnom, GAMMA by XTI and VG
                    const double rT = T / Tn;
                    k.rco *= std::pow(rT, c->get("CN", q.isNPN ? 2.42 : 2.2));
                    k.vo *= std::pow(rT, c->get("D", q.isNPN ? 0.87 : 0.52));
                    k.gamma *= std::pow(rT, c->get("XTI", 3.0)) * std::exp(-c->get("VG", 1.206) * (1.0 - rT) / vt(T));
                }
                const bool tbf = c->params.count("TBF1") || c->params.count("TBF2"), tbr = c->params.count("TBR1") || c->params.count("TBR2");
                q.betaF *= tbf ? tc("TBF1", "TBF2") : bfac;
                q.betaR *= tbr ? tc("TBR1", "TBR2") : bfac;
                if (tlevc == 1.0) {
                    k.cje *= 1.0 + c->get("CTE", 0.0) * dt; k.vje -= c->get("TVJE", 0.0) * dt;
                    k.cjc *= 1.0 + c->get("CTC", 0.0) * dt; k.vjc -= c->get("TVJC", 0.0) * dt;
                    k.cjs *= 1.0 + c->get("CTS", 0.0) * dt; k.vjs -= c->get("TVJS", 0.0) * dt;
                } else {
                    k.cje *= junctionAt(k.vje, k.mje, Tn, T);
                    k.cjc *= junctionAt(k.vjc, k.mjc, Tn, T);
                    k.cjs *= junctionAt(k.vjs, k.mjs, Tn, T);
                }
            }
            if (!(k.mje < 1.0 && k.mjc < 1.0 && k.mjs < 1.0 && k.vje > 0.0 && k.vjc > 0.0 && k.vjs > 0.0 &&
                  k.fc < 1.0 && k.xcjc >= 0.0 && k.xcjc <= 1.0 && k.nf > 0.0 && k.nr > 0.0 &&
                  k.rb >= 0.0 && k.rc >= 0.0 && k.re >= 0.0 && q.betaF > 0.0 && q.betaR > 0.0))
                throw std::runtime_error("model " + c->name + ": needs MJE/MJC/MJS < 1, VJE/VJC/VJS > 0, FC < 1, "
                                         "0 <= XCJC <= 1, NF/NR/BF/BR > 0, RB/RC/RE >= 0");
        }
        for (auto& o : blk.osdiDevices) {
            if (o.model) continue;  // a BSIM4 card, bridged above
            const ModelCard* c = sensed(card(o.modelName), o.instanceName);
            sensList(blk, o.instanceName, *c);
            const osdi::Descriptor* d = osdi::findModule(c->type);
            if (!d) throw std::runtime_error("model " + c->name + ": type " + c->type +
                                             " is not a loaded OSDI module (load it with .osdi <file>)");
            auto& m = osdiModels_[toUpper(c->name)];
            if (!m) m = std::make_shared<osdi::Model>(d, c->name, c->params);
            o.model = m;
        }
        for (auto& j : blk.jfets) if (const ModelCard* c = sensed(card(j.modelName), j.instanceName)) {
            sensList(blk, j.instanceName, *c);
            if (j.mes) {  // ngspice mes*.c (LEVEL 1, Statz): no temperature dependence, its junctions at REFTEMP's kT/q
                if (c->type != "NMF" && c->type != "PMF")
                    throw std::runtime_error(j.instanceName + ": model " + c->name + " is " + c->type + ", not an NMF/PMF model");
                if (c->get("LEVEL", 1.0) != 1.0)
                    throw std::runtime_error("model " + c->name + ": MESFET LEVEL 2-4 (MESA) is not supported yet");
                supported(c, {"VTO", "VT0", "ALPHA", "BETA", "LAMBDA", "B", "RD", "RS", "CGS", "CGD", "PB", "IS", "FC", "KF", "AF"});
                bounded(c, {"BETA", "LAMBDA", "B", "RD", "RS", "CGS", "CGD", "IS", "KF"}, {"ALPHA", "AF"});
                if (!(c->get("PB", 1.0) > 0.5)) throw std::runtime_error("model " + c->name + ": PB must be > 0.5 (the Statz charge)");
                j.isNChannel = c->type == "NMF";
                j.thermalVoltage_V_T_V = devtemp::vt(devtemp::REFTEMP);
                j.beta = c->get("BETA", 2.5e-3);
                j.Vto = c->params.count("VT0") ? c->get("VT0", -2.0) : c->get("VTO", -2.0);
                j.lambda = c->get("LAMBDA", 0.0);
                auto& k = j.card;
                k.n = 1.0; k.b = c->get("B", 0.3); k.alpha = c->get("ALPHA", 2.0); k.is = c->get("IS", 1e-14);
                k.pb = c->get("PB", 1.0); k.fc = c->get("FC", 0.5);  // FC: mestemp's f1..f3, which no equation reads
                k.cgs = c->get("CGS", 0.0); k.cgd = c->get("CGD", 0.0);
                k.kf = c->get("KF", 0.0); k.af = c->get("AF", 1.0);
                continue;
            }
            if (c->type != "NJF" && c->type != "PJF" && c->type != "NJFET" && c->type != "PJFET")
                throw std::runtime_error(j.instanceName + ": model " + c->name + " is " + c->type + ", not an NJF/PJF model");
            supported(c, {"BETA", "VTO", "LAMBDA", "IS", "N", "B", "CGS", "CGD", "PB", "FC", "KF", "AF", "RD", "RS",
                          "TNOM", "XTI", "EG", "VTOTC", "BETATCE", "TCV", "BEX"});
            bounded(c, {"BETA", "LAMBDA", "IS", "CGS", "CGD", "KF", "RD", "RS", "XTI"}, {"N", "B", "PB", "EG"});
            // ngspice jfettemp: IS, PB and CGS/CGD by the junction laws; VTO by VTOTC (else TCV), BETA by BETATCE (else BEX)
            const double T = ambientK_, Tn = c->get("TNOM", tnomC_) + 273.15, dT = T - Tn;
            j.thermalVoltage_V_T_V = devtemp::vt(T);
            j.isNChannel = (c->type == "NJF" || c->type == "NJFET");
            j.beta   = c->get("BETA", 1e-4) * (c->params.count("BETATCE") ? std::pow(1.01, c->get("BETATCE", 0.0) * dT)
                                                                                 : std::pow(T / Tn, c->get("BEX", 0.0)));
            j.Vto    = c->params.count("VTOTC") ? c->get("VTO", -2.0) + c->get("VTOTC", 0.0) * dT
                                                : c->get("VTO", -2.0) - c->get("TCV", 0.0) * dT;
            j.lambda = c->get("LAMBDA", 0.0);
            auto& k = j.card;
            k.n = c->get("N", 1.0); k.b = c->get("B", 1.0);
            // ngspice (probed): N divides the band-gap term but not XTI, and XTI defaults to 0
            k.is = c->get("IS", 1e-14) * std::exp((T / Tn - 1) * c->get("EG", 1.11) / (k.n * devtemp::vt(T))) *
                   std::pow(T / Tn, c->get("XTI", 0.0));
            k.pb = c->get("PB", 1.0);
            const double capf = devtemp::junctionAt(k.pb, 0.5, Tn, T);
            k.cgs = c->get("CGS", 0.0) * capf; k.cgd = c->get("CGD", 0.0) * capf; k.fc = c->get("FC", 0.5);
            k.kf = c->get("KF", 0.0); k.af = c->get("AF", 1.0);
            if (!(k.fc < 1.0 && k.pb > j.Vto))
                throw std::runtime_error("model " + c->name + ": needs FC < 1 and PB > VTO");
        }
        {
            const double ts = result_.tranStep;
            auto edges = [&](auto& s) {
                auto dflt = [](double& x, double v) { if (std::isnan(x)) x = v; };
                if (s.type == "EXP") {
                    auto& w = s.wf;
                    dflt(w[2], ts); dflt(w[3], ts); dflt(w[4], w[2] + ts); dflt(w[5], ts);
                }
                // SFFM: ngspice's TSTOP defaults, then the index held to [0, FC/FM]
                // ponytail: with no .tran card (a transient requested through the API) FC / FM stay NaN, as EXP's
                if (s.type == "SFFM") {
                    auto& w = s.wf;
                    if (result_.tranStop > 0.0) { dflt(w[2], 5.0 / result_.tranStop); dflt(w[4], 500.0 / result_.tranStop); }
                    dflt(w[3], 90.0);
                    if (!std::isnan(w[2]) && !std::isnan(w[4])) w[3] = std::clamp(w[3], 0.0, w[2] / w[4]);
                }
                if (s.type != "PULSE" || !(ts > 0.0)) return;
                if (s.pulse_tr <= 0.0) s.pulse_tr = ts;
                if (s.pulse_tf <= 0.0) s.pulse_tf = ts;
            };
            for (auto& v : blk.voltageSources) edges(v);
            for (auto& i : blk.currentSources) edges(i);
        }
    };
    // Errors found here belong to a card: the .model card for "model X: ...",
    // else the device being resolved (resolveAt_), not the end of the file.
    try {
        resolveAt_ = "";
        resolveBlock(result_.netlist.globalBlock);
        for (auto& [name, blk] : result_.netlist.blockDefinitions) { resolveAt_ = name; resolveBlock(blk); }
    } catch (const std::exception& e) {
        const std::string what = e.what();
        int at = lines.back().first;
        if (what.rfind("model ", 0) == 0) {
            const auto m = cardLine_.find(".MODEL/" + toUpper(what.substr(6, what.find(':') - 6)));
            if (m != cardLine_.end()) at = m->second;
        } else if (const auto d = cardLine_.find(resolveAt_ + "/" + toUpper(what.substr(0, what.find(':'))));
                   d != cardLine_.end()) at = d->second;
        return fail(at, what);
    }

    // Merge model cards into netlist
    for (auto& [name, card] : modelCards_) {
        result_.netlist.addModelCard(card);
        result_.modelsCreated++;
    }

    // Initialize global state; instance states again (resolving moved BSIM4 cards
    // from the MOSFET list to the OSDI list after the instances were placed)
    result_.netlist.globalState.initialize(result_.netlist.globalBlock);
    for (size_t k = 0; k < result_.netlist.instances.size() && k < result_.netlist.instanceStates.size(); ++k)
        if (auto it = result_.netlist.blockDefinitions.find(result_.netlist.instances[k].blockName);
            it != result_.netlist.blockDefinitions.end())
            result_.netlist.instanceStates[k].initialize(it->second);
    if (const char* b = std::getenv("TREESPICE_BYPASS"); b && std::atoi(b) == 0) result_.netlist.environment.bypass = false;
    try {  // OSDI instances: parameters, setup_instance, node collapsing
        result_.netlist.instantiateOsdi(result_.netlist.environment.ambient_temp_K, result_.netlist.environment.bypass);
    } catch (const std::exception& e) {  // "<BLOCK>#<k>.<name>: ..." or "<name>: ...": the device's line
        const std::string what = e.what(), who = what.substr(0, what.find(':'));
        const size_t hash = who.find('#'), dot = who.rfind('.');
        const std::string blk = hash == std::string::npos ? "" : who.substr(0, hash);
        const auto it = cardLine_.find(blk + "/" + toUpper(dot == std::string::npos ? who : who.substr(dot + 1)));
        return fail(it == cardLine_.end() ? lines.back().first : it->second, what);
    }

    return result_;
}

// ============================================================================
// Element Parsers
// ============================================================================

void SPICEParser::parseResistor(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 4);
    Resistor r;
    r.name = tok[0];
    r.nodeTerminal1 = nodeIndex(tok[1]);
    r.nodeTerminal2 = nodeIndex(tok[2]);
    r.resistance_ohms = NAN;  // NAN: from the model (resolveBlock)
    size_t i = 3;  // "R1 a b r=1k" (IHP cap_cmim), "R1 a b 1k", "R1 a b [1k] MODEL L= W="
    if (const std::string k = toUpper(tok[3]); (k == "R" || k == "RESISTANCE") && tok.size() > 4) r.resistance_ohms = parseValue(tok[i = 4]), ++i;
    else if (isValue(tok[3])) r.resistance_ohms = parseValue(tok[i++]);
    if (i < tok.size() && modelNames_.count(toUpper(tok[i]))) r.model_name = tok[i++];
    if (std::isnan(r.resistance_ohms) && r.model_name.empty()) r.resistance_ohms = parseValue(tok[3]);  // the error for a bad value
    r.m = mult_;
    if (r.resistance_ohms == 0.0) {  // an exact short: a 0 V branch (negative R is legal SPICE)
        blk.voltageSources.push_back({r.nodeTerminal1, r.nodeTerminal2, 0.0});
        return;
    }

    // tokenize() turns "KEY=val" into ["KEY", "val"]
    for (; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (i + 1 >= tok.size()) throw std::runtime_error(tok[0] + ": parameter " + upper + " has no value");
        if (upper == "W")        r.width_m  = parseValue(tok[++i]);
        else if (upper == "L")   r.length_m = parseValue(tok[++i]);
        else if (upper == "M")   r.m = parseValue(tok[++i]);
        else if (upper == "AC")  { r.ac_ohms = parseValue(tok[++i]); if (r.ac_ohms == 0.0) throw std::runtime_error(tok[0] + ": AC must not be 0"); }
        else if (upper == "TC1") r.tc1 = parseValue(tok[++i]);
        else if (upper == "TC2") r.tc2 = parseValue(tok[++i]);
        else if (upper == "TC1") { r.tc1 = parseValue(tok[++i]); r.tc1Given = true; }  // R(T) = R (1 + tc1 (T-TNOM) + tc2 (T-TNOM)^2)
        else if (upper == "TC2") { r.tc2 = parseValue(tok[++i]); r.tc2Given = true; }
        else if (upper == "TC") {  // TC=tc1[,tc2] (the tokenizer split the comma)
            r.tc1 = parseValue(tok[++i]); r.tc1Given = true;
            if (i + 1 < tok.size() && isValue(tok[i + 1])) { r.tc2 = parseValue(tok[++i]); r.tc2Given = true; }
        }
        else if (upper == "TEMP") r.inst.temp = parseValue(tok[++i]);
        else if (upper == "DTEMP") r.inst.dtemp = parseValue(tok[++i]);
        else throw std::runtime_error(tok[0] + ": parameter " + upper + " is not supported yet");
    }
    if (!(r.m > 0.0)) throw std::runtime_error(tok[0] + ": M must be > 0");
    blk.resistors.push_back(r);
}

void SPICEParser::parseCapacitor(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 4);
    Capacitor c;
    c.name = tok[0];
    c.nodePlate1 = nodeIndex(tok[1]);
    c.nodePlate2 = nodeIndex(tok[2]);
    // C a b <value> [model] | C a b <model> (semiconductor capacitor), then KEY val pairs:
    // an even count of words after the nodes means value + model; odd, value or model
    // or C a b c=<value> ... (ngspice's keyword form)
    size_t i = 3;
    const char c0 = tok[3][0];
    const bool keyword = (toUpper(tok[3]) == "C" || toUpper(tok[3]) == "CAPACITANCE") && tok.size() > 4;
    const bool valueFirst = !keyword && ((tok.size() - 3) % 2 == 0 || std::isdigit((unsigned char)c0) || c0 == '.' || c0 == '-' || c0 == '+');
    c.capacitance_farads = valueFirst ? parseValue(tok[i++]) : NAN;  // NAN: from the model (resolveBlock)
    if (!keyword && (tok.size() - i) % 2 == 1) c.model_name = tok[i++];

    // tokenize() turns "KEY=val" into ["KEY", "val"]. IC= applies with UIC.
    for (; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (i + 1 >= tok.size()) throw std::runtime_error(tok[0] + ": parameter " + upper + " has no value");
        if (upper == "C" || upper == "CAPACITANCE") c.capacitance_farads = parseValue(tok[++i]);
        else if (upper == "W") c.width_m  = parseValue(tok[++i]);
        else if (upper == "L") c.length_m = parseValue(tok[++i]);
        else if (upper == "M") c.m        = parseValue(tok[++i]);
        else if (upper == "SCALE") c.scale = parseValue(tok[++i]);
        else if (upper == "TC1") { c.tc1 = parseValue(tok[++i]); c.tc1Given = true; }
        else if (upper == "TC2") { c.tc2 = parseValue(tok[++i]); c.tc2Given = true; }
        else if (upper == "TEMP") c.inst.temp = parseValue(tok[++i]);
        else if (upper == "DTEMP") c.inst.dtemp = parseValue(tok[++i]);
        else if (upper == "IC") { c.ic_V = parseValue(tok[++i]); c.icGiven = true; }
        else throw std::runtime_error(tok[0] + ": parameter " + upper + " is not supported yet");
    }
    blk.capacitors.push_back(c);
}

void SPICEParser::parseInductor(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 4);
    Inductor ind;
    ind.name = tok[0];
    ind.nodeCoil1 = nodeIndex(tok[1]);
    ind.nodeCoil2 = nodeIndex(tok[2]);
    // L a b <value> | L a b l=<value> (ngspice's keyword form)
    const bool keyword = (toUpper(tok[3]) == "L" || toUpper(tok[3]) == "INDUCTANCE") && tok.size() > 4;
    ind.inductance_henries = parseValue(tok[keyword ? 4 : 3]) / mult_;
    // IC= (initial current) applies with UIC; anything else used to be ignored silently
    for (size_t i = keyword ? 5 : 4; i < tok.size(); ++i) {
        if (toUpper(tok[i]) != "IC" || i + 1 >= tok.size())
            throw std::runtime_error(tok[0] + ": unsupported inductor parameter '" + tok[i] + "'");
        ind.ic_A = parseValue(tok[++i]);
        ind.icGiven = true;
    }

    // Track inductor by name for K-statement resolution
    inductorIndex_[toUpper(tok[0])] = static_cast<int>(blk.inductors.size());

    blk.inductors.push_back(ind);
}

void SPICEParser::parseMutualInd(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() != 4)  // a core model or a third inductor used to be dropped silently
        throw std::runtime_error(tok[0] + ": only K <L1> <L2> <k> (linear, two inductors) is supported");
    MutualInductor mi;
    auto it1 = inductorIndex_.find(toUpper(tok[1]));
    auto it2 = inductorIndex_.find(toUpper(tok[2]));
    if (it1 == inductorIndex_.end() || it2 == inductorIndex_.end())
        throw std::runtime_error(tok[0] + " references an undefined inductor");

    mi.inductor1_index = it1->second;
    mi.inductor2_index = it2->second;
    mi.couplingCoefficient_k = parseValue(tok[3]);
    blk.mutualInductors.push_back(mi);
}

// V<name> <n+> <n-> [DC <val>] [AC <mag> [<phase>]] [PULSE(...)] [SIN(...)]
void SPICEParser::parseVoltSrc(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 3);
    VoltageSource vs;
    vs.name = toUpper(tok[0]);
    vs.nodePositive = nodeIndex(tok[1]);
    vs.nodeNegative = nodeIndex(tok[2]);
    parseSourceSpec(tok, vs, vs.voltage_V);
    blk.voltageSources.push_back(vs);
}

// Shared V/I spec. If no DC value is given, the DC value is the waveform's
// t=0 value (SPICE rule), so every DC code path can keep using voltage_V.
template <class Src>
void SPICEParser::parseSourceSpec(const std::vector<std::string>& tok, Src& s, double& dc) const {
    bool hasDC = false;
    for (size_t i = 3; i < tok.size();) {
        std::string kw = toUpper(tok[i]);
        if (kw == "DC") {
            hasDC = true;  // bare "DC" means 0
            if (i + 1 < tok.size() && isValue(tok[i + 1])) { dc = parseValue(tok[i + 1]); i += 2; }
            else ++i;
        } else if (kw == "AC") {
            ++i;
            s.ac_mag = 1.0;  // SPICE: bare "AC" means unit magnitude
            if (i < tok.size() && isValue(tok[i])) s.ac_mag = parseValue(tok[i++]);
            if (i < tok.size() && isValue(tok[i])) s.ac_phase = parseValue(tok[i++]);
        } else if (kw == "PULSE") {
            s.type = "PULSE";
            parsePulse(tok, i + 1, s);
            break;  // waveform consumes the rest of the card
        } else if (kw == "SIN") {
            s.type = "SINE";  // canonical name used by the stampers
            parseSine(tok, i + 1, s);
            break;
        } else if (kw == "PWL") {
            s.type = "PWL";
            parsePwl(tok, i + 1, s);
            break;
        } else if (kw == "EXP" || kw == "SFFM" || kw == "AM") {
            s.type = kw;
            // ngspice order. EXP's missing values are NaN until .tran is known (resolveBlock): TD1,
            // TAU1, TAU2 take TSTEP and TD2 TD1+TSTEP.
            // ponytail: with no .tran card (a transient requested through the API) they stay NaN.
            // SFFM (ngspice 47): VO VA [FC MDI FM TD PHASEM PHASEC]; FC, MDI, FM default 5/TSTOP, 90, 500/TSTOP (NaN until
            // resolveBlock knows .tran; an FM of 0 is not given)
            const size_t lo = kw == "AM" ? 5 : 2, hi = kw == "EXP" || kw == "AM" ? 6 : kw == "SFFM" ? 8 : 5;
            const double nan = std::nan("");
            s.wf = kw == "EXP" ? std::vector<double>{0, 0, nan, nan, nan, nan}
                 : kw == "SFFM" ? std::vector<double>{0, 0, nan, nan, nan, 0, 0, 0}
                 : std::vector<double>(6, 0.0);
            size_t n = 0;
            for (++i; i < tok.size() && n < hi && isValue(tok[i]); ++i, ++n) s.wf[n] = parseValue(tok[i]);
            if (n < lo) throw std::runtime_error(tok[0] + ": " + kw + " needs at least " + std::to_string(lo) + " values");
            if (i < tok.size()) throw std::runtime_error(tok[0] + ": " + kw + " takes at most " + std::to_string(hi) + " values");
            if (kw == "EXP")  // ngspice: a 0 TD1, TAU1, TD2 or TAU2 is "not given"
                for (size_t k = 2; k < 6; ++k) if (s.wf[k] == 0.0) s.wf[k] = nan;
            if (kw == "SFFM" && s.wf[4] == 0.0) s.wf[4] = nan;
            if (!hasDC) { dc = kw == "EXP" ? s.wf[0] : 0.0; return; }  // ngspice's DC value of SFFM and AM is 0, not their t = 0 value
            break;
        } else {
            dc = parseValue(tok[i++]);  // bare value, e.g. "V1 1 0 5"
            hasDC = true;
        }
    }
    if (s.type.empty()) s.type = "DC";
    if (!hasDC) dc = waveformValue(s, 0.0, 0.0);
}

// PULSE V1 V2 TD TR TF PW PER
template <class Src>
void SPICEParser::parsePulse(const std::vector<std::string>& tok, size_t idx, Src& s) const {
    double* f[] = {&s.pulse_v1, &s.pulse_v2, &s.pulse_td, &s.pulse_tr,
                   &s.pulse_tf, &s.pulse_pw, &s.pulse_per};
    size_t n = 0;
    for (; n < 7 && idx + n < tok.size(); ++n) *f[n] = parseValue(tok[idx + n]);
    if (n < 6) s.pulse_pw = 1e99;  // SPICE default PW = TSTOP: never falls
    // TR/TF = 0 get SPICE's TSTEP default in parse() once .tran is known
}

// PWL t1 v1 t2 v2 ... [TD delay] [R repeat-from] (tokenize() splits KEY=val)
template <class Src>
void SPICEParser::parsePwl(const std::vector<std::string>& tok, size_t idx, Src& s) const {
    for (; idx < tok.size(); ++idx) {
        const std::string k = toUpper(tok[idx]);
        if ((k == "TD" || k == "R") && idx + 1 < tok.size()) {
            (k == "TD" ? s.pwl_td : s.pwl_r) = parseValue(tok[++idx]);
            continue;
        }
        if (idx + 1 >= tok.size()) throw std::runtime_error(tok[0] + ": PWL time without a value");
        s.pwl_t.push_back(parseValue(tok[idx]));
        s.pwl_v.push_back(parseValue(tok[++idx]));
    }
    if (s.pwl_t.empty()) throw std::runtime_error(tok[0] + ": PWL needs at least one (time, value) pair");
    // Equal times are a step (ngspice warns and steps). Decreasing times fail closed (ngspice
    // warns and clamps them).
    for (size_t k = 1; k < s.pwl_t.size(); ++k)
        if (s.pwl_t[k] < s.pwl_t[k - 1])
            throw std::runtime_error(tok[0] + ": decreasing PWL times are not supported");
    if (s.pwl_r >= 0.0 && std::find(s.pwl_t.begin(), s.pwl_t.end(), s.pwl_r) == s.pwl_t.end())
        throw std::runtime_error(tok[0] + ": PWL R= must be one of its times");
}

// SIN VO VA FREQ TD THETA PHASE
template <class Src>
void SPICEParser::parseSine(const std::vector<std::string>& tok, size_t idx, Src& s) const {
    double* f[] = {&s.sine_vo, &s.sine_va, &s.sine_freq, &s.sine_td,
                   &s.sine_theta, &s.sine_phase};
    for (size_t n = 0; n < 6 && idx + n < tok.size(); ++n) *f[n] = parseValue(tok[idx + n]);
}

// I<name> <n+> <n-> [DC <val>] [AC <mag> [<phase>]] [PULSE(...)] [SIN(...)]
void SPICEParser::parseCurrSrc(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 3);
    CurrentSource cs;
    cs.name = toUpper(tok[0]);
    cs.nodePositive = nodeIndex(tok[1]);
    cs.nodeNegative = nodeIndex(tok[2]);
    parseSourceSpec(tok, cs, cs.current_A);
    if (mult_ != 1.0) {  // m parallel sources: every amplitude scales
        for (double* a : {&cs.current_A, &cs.ac_mag, &cs.pulse_v1, &cs.pulse_v2, &cs.sine_vo, &cs.sine_va}) *a *= mult_;
        for (double& v : cs.pwl_v) v *= mult_;
    }
    blk.currentSources.push_back(cs);
}

// D<name> <n+> <n-> <modelName> [area] [AREA=] [M=] [OFF] [IC=]   (params from .model in parse())
void SPICEParser::parseDiode(const std::vector<std::string>& tok0, TensorBlock& blk) {
    requireFields(tok0, 4);
    Diode d;
    std::vector<std::string> tok = tok0;  // D a k [tj] model ...: a third node before the model is the thermal one
    if (tok.size() > 4 && !modelNames_.count(toUpper(tok[3])) && modelNames_.count(toUpper(tok[4]))) {
        d.thermal = nodeIndex(tok[3]);
        tok.erase(tok.begin() + 3);
    }
    std::vector<std::string> rest(tok.begin(), tok.begin() + 4);  // PJ= is the sidewall perimeter; M= also scales it
    for (size_t i = 4; i < tok.size(); ++i) {
        const std::string k = toUpper(tok[i]);
        if (k == "THERMAL") { d.thermalFlag = true; continue; }
        if ((k == "W" || k == "L" || k == "LM" || k == "LP" || k == "WM" || k == "WP") && i + 1 < tok.size() && isValue(tok[i + 1])) {
            const double v = parseValue(tok[++i]);
            (k == "W" ? d.w : k == "L" ? d.l : k == "LM" ? d.lm : k == "LP" ? d.lp : k == "WM" ? d.wm : d.wp) = v;
            continue;
        }
        if (k == "PJ" && i + 1 < tok.size() && isValue(tok[i + 1])) { d.pj = parseValue(tok[++i]); d.pjGiven = true; continue; }
        if (k == "AREA" || (i == 4 && isValue(tok[i]))) d.areaGiven = true;
        if (k == "M" && i + 1 < tok.size() && isValue(tok[i + 1])) d.mult = parseValue(tok[i + 1]);
        rest.push_back(tok[i]);
    }
    d.area = instanceArea(rest, 4, &d.inst, &d.off, d.ic, 1);
    d.instanceName = tok[0];
    d.anode = nodeIndex(tok[1]);
    d.cathode = nodeIndex(tok[2]);
    d.modelName = tok[3];
    blk.diodes.push_back(d);
}

// M<name> <nd> <ng> <ns> <nb> <modelName> [W=] [L=] [AD=] [AS=] [PD=] [PS=] [NRD=] [NRS=] [NF=] [M=]
void SPICEParser::parseMosfet(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 5);
    Mosfet m;
    m.instanceName = tok[0];
    m.drain  = nodeIndex(tok[1]);
    m.gate   = nodeIndex(tok[2]);
    m.source = nodeIndex(tok[3]);
    auto isModel = [&](const std::string& t) {  // a .model card, or a binned family NAME.0, NAME.1, ...
        const std::string mu = toUpper(t);
        const auto bin = modelNames_.lower_bound(mu + ".");
        return modelNames_.count(mu) || (bin != modelNames_.end() && bin->compare(0, mu.size() + 1, mu + ".") == 0);
    };
    size_t mi = 5;  // the model name: after 4 nodes, or after up to 3 more (BSIM-SOI P, B, T); a VDMOS has 3 (or 5)
    m.threeTerm = isModel(tok[4]);
    if (m.threeTerm) { mi = 4; m.body = 0; } else m.body = nodeIndex(tok[4]);
    while (mi < 8 && mi + 1 < tok.size() && !isModel(tok[mi])) ++mi;
    if (mi >= tok.size() || !isModel(tok[mi]))
        throw std::runtime_error(tok[0] + ": expected 4 nodes (d g s b), then a model name; '" + tok.back() +
                                 "' is not a .model card");
    for (size_t k = 5; k < mi; ++k) m.soiNodes.push_back(nodeIndex(tok[k]));
    m.modelName = tok[mi];
    m.w = m.l = 100e-6;  // SPICE DEFW / DEFL (not scaled)

    // tokenize() turns "W=4u" into ["W", "4u"]. AD/AS/PD/PS act through CJ/CJSW;
    // NRD/NRS only through RSH, which the model check rejects. SA/SB/SD and NF
    // reach BSIM4 (level 54); level 1 rejects NF != 1 when its card is resolved.
    // .option scale multiplies the lengths (areas by scale^2), as in ngspice.
    for (size_t i = mi + 1; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (upper == "OFF") { m.off = true; continue; }
        if (upper == "THERMAL") { m.thermal = true; continue; }
        if (upper == "IC") {  // IC=vds[,vgs[,vbs]]
            int n = 0;
            while (n < 3 && i + 1 < tok.size() && isValue(tok[i + 1])) m.ic[n++] = parseValue(tok[++i]);
            if (n == 0) throw std::runtime_error(tok[0] + ": IC needs a value");
            continue;
        }
        if (upper == "ICVDS" || upper == "ICVGS" || upper == "ICVBS") {
            if (i + 1 >= tok.size()) throw std::runtime_error(tok[0] + ": parameter " + upper + " has no value");
            m.ic[upper == "ICVDS" ? 0 : upper == "ICVGS" ? 1 : 2] = parseValue(tok[++i]);
            continue;
        }
        if (i + 1 >= tok.size()) throw std::runtime_error(tok[0] + ": parameter " + upper + " has no value");
        const double s = scale_;
        if (upper == "W")        m.w = parseValue(tok[++i]) * s;
        else if (upper == "L")   m.l = parseValue(tok[++i]) * s;
        else if (upper == "AD")  m.geo.ad = parseValue(tok[++i]) * s * s;
        else if (upper == "AS")  m.geo.as = parseValue(tok[++i]) * s * s;
        else if (upper == "PD")  m.geo.pd = parseValue(tok[++i]) * s;
        else if (upper == "PS")  m.geo.ps = parseValue(tok[++i]) * s;
        else if (upper == "NRD") m.geo.nrd = parseValue(tok[++i]);
        else if (upper == "NRS") m.geo.nrs = parseValue(tok[++i]);
        else if (upper == "SA")  m.geo.sa = parseValue(tok[++i]) * s;
        else if (upper == "SB")  m.geo.sb = parseValue(tok[++i]) * s;
        else if (upper == "SD")  m.geo.sd = parseValue(tok[++i]) * s;
        else if (upper == "NF")  m.extra.nf = (int)std::lround(parseValue(tok[++i]));
        else if (upper == "TEMP" || upper == "DTEMP") { (upper == "TEMP" ? m.inst.temp : m.inst.dtemp) = parseValue(tok[++i]); continue; }
        else if (upper == "M" || upper == "MULT") m.extra.mult = parseValue(tok[++i]);
        else { m.osdiParams[upper] = parseValue(tok[++i]); continue; }  // checked where the model is known
        if (upper != "M" && upper != "MULT") {  // as written (scaled): a BSIM4 device gets exactly these
            const std::map<std::string, double> v = {{"W", m.w}, {"L", m.l}, {"AD", m.geo.ad}, {"AS", m.geo.as},
                {"PD", m.geo.pd}, {"PS", m.geo.ps}, {"NRD", m.geo.nrd}, {"NRS", m.geo.nrs}, {"SA", m.geo.sa},
                {"SB", m.geo.sb}, {"SD", m.geo.sd}, {"NF", (double)m.extra.nf}};
            m.given[upper] = v.at(upper);
        }
    }
    if (!(m.l > 0.0) || !(m.w > 0.0)) throw std::runtime_error(tok[0] + ": W and L must be > 0");
    blk.mosfets.push_back(m);
}

// Q<name> <nc> <nb> <ne> [<ns>] <modelName> [area] [OFF]   (params from .model in parse())
void SPICEParser::parseBJT(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 5);
    BJT q;
    q.instanceName = tok[0];
    q.nodeCollector = nodeIndex(tok[1]);
    q.base = nodeIndex(tok[2]);
    q.emitter = nodeIndex(tok[3]);

    // Optional substrate and thermal nodes: c b e [s [t]] model, the model being the first of tok[4..6]
    // that names a .model card (all names are known after the first pass). No substrate node: ground.
    auto model = [&](size_t i) { return tok.size() > i && modelNames_.count(toUpper(tok[i])); };
    const size_t modelIdx = model(4) || !(model(5) || model(6)) ? 4 : model(5) ? 5 : 6;
    if (modelIdx >= 5) q.substrate = nodeIndex(tok[4]);
    if (modelIdx == 6) q.thermal = nodeIndex(tok[5]);
    q.modelName = tok[modelIdx];
    q.area = instanceArea(tok, modelIdx + 1, &q.inst, &q.off, q.ic, 2);
    for (size_t i = modelIdx + 1; i + 1 < tok.size(); ++i)
        if (toUpper(tok[i]) == "M") q.m = parseValue(tok[i + 1]);
    blk.bjts.push_back(q);
}

// S<name> <n+> <n-> <nc+> <nc-> <model> [ON|OFF]   W<name> <n+> <n-> <Vctrl> <model> [ON|OFF]
void SPICEParser::parseSwitch(const std::vector<std::string>& tok, TensorBlock& blk) {
    const bool w = toUpper(tok[0])[0] == 'W';
    const size_t m = w ? 4 : 5;
    requireFields(tok, m + 1);
    Switch s;
    s.name = toUpper(tok[0]);
    s.nodePos = nodeIndex(tok[1]);
    s.nodeNeg = nodeIndex(tok[2]);
    if (w) s.ctrlName = toUpper(tok[3]);
    else { s.ctrlPos = nodeIndex(tok[3]); s.ctrlNeg = nodeIndex(tok[4]); }
    s.modelName = tok[m];
    if (tok.size() > m + 2) throw std::runtime_error(tok[0] + ": expected the model and ON or OFF");
    if (tok.size() == m + 2) {
        const std::string k = toUpper(tok[m + 1]);
        if (k != "ON" && k != "OFF") throw std::runtime_error(tok[0] + ": parameter " + k + " is not supported yet");
        s.init = k == "ON";
    }
    blk.switches.push_back(s);
}

// J<name> <nd> <ng> <ns> <modelName>   (params from .model in parse())
void SPICEParser::parseJFET(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 5);
    JFET j;
    j.area = instanceArea(tok, 5, nullptr, &j.off, j.ic, 2) * mult_;  // ngspice jfetload: M on every stamp, as AREA
    j.instanceName = tok[0];
    j.drain  = nodeIndex(tok[1]);
    j.gate   = nodeIndex(tok[2]);
    j.source = nodeIndex(tok[3]);
    j.modelName = tok[4];
    j.m = mult_;
    blk.jfets.push_back(j);
}

// U<name> <n1> <n2> <n3> <model> L=<length> [N=<lumps>]   (uniform distributed RC line, ngspice urc)
// O<name> <p1> <n1> <p2> <n2> <model> [V1=v I1=i V2=v I2=i | IC=v1,i1,v2,i2]   (ngspice LTRA)
void SPICEParser::parseLossyLine(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 6);
    LossyLine o;
    o.name = tok[0];
    o.p1 = nodeIndex(tok[1]); o.n1 = nodeIndex(tok[2]); o.p2 = nodeIndex(tok[3]); o.n2 = nodeIndex(tok[4]);
    o.modelName = tok[5];
    for (size_t i = 6; i < tok.size(); ++i) {
        const std::string k = toUpper(tok[i]);
        if (i + 1 >= tok.size()) throw std::runtime_error(tok[0] + ": parameter " + k + " has no value");
        if (k == "IC") {
            if (i + 4 >= tok.size()) throw std::runtime_error(tok[0] + ": IC takes v1, i1, v2, i2");
            for (double& v : o.ic) v = parseValue(tok[++i]);
        } else if (k == "V1") o.ic[0] = parseValue(tok[++i]);
        else if (k == "I1") o.ic[1] = parseValue(tok[++i]);
        else if (k == "V2") o.ic[2] = parseValue(tok[++i]);
        else if (k == "I2") o.ic[3] = parseValue(tok[++i]);
        else throw std::runtime_error(tok[0] + ": parameter " + k + " is not supported yet");
        o.icGiven = true;
    }
    blk.lossyLines.push_back(o);
}

void SPICEParser::parseURC(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 5);
    TensorBlock::URC u;
    u.name = tok[0];
    u.n1 = nodeIndex(tok[1]); u.n2 = nodeIndex(tok[2]); u.n3 = nodeIndex(tok[3]);
    u.model = tok[4];
    for (size_t i = 5; i < tok.size(); ++i) {
        const std::string k = toUpper(tok[i]);
        if (i + 1 >= tok.size()) throw std::runtime_error(tok[0] + ": parameter " + k + " has no value");
        if (k == "L") u.length = parseValue(tok[++i]) * scale_;
        else if (k == "N") u.lumps = (int)std::lround(parseValue(tok[++i]));
        else throw std::runtime_error(tok[0] + ": parameter " + k + " is not supported yet");
    }
    if (!(u.length > 0.0)) throw std::runtime_error(tok[0] + ": L must be > 0");
    if (u.lumps < 0 || u.lumps == 1 || u.lumps == 2) throw std::runtime_error(tok[0] + ": N must be at least 3");
    blk.urcs.push_back(u);
}

// N<name> <terminal>... <modelName> [<param> <value>]...   (OSDI device, as ngspice)
// The model name is the first field naming a .model card (all are known after
// the first pass), so the terminals are numbered in order like any element.
void SPICEParser::parseOsdiDevice(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 3);
    size_t m = 1;
    while (m < tok.size() && !modelNames_.count(toUpper(tok[m]))) ++m;
    if (m == tok.size()) throw std::runtime_error(tok[0] + ": no field names a .model card");
    if (m == 1) throw std::runtime_error(tok[0] + ": no terminals before the model name");
    if ((tok.size() - m - 1) % 2) throw std::runtime_error(tok[0] + ": instance parameters must be name=value pairs");
    OsdiDevice o;
    o.instanceName = tok[0];
    o.modelName = tok[m];
    for (size_t i = 1; i < m; ++i) o.nodes.push_back(nodeIndex(tok[i]));
    for (size_t i = m + 1; i + 1 < tok.size(); i += 2) o.params.push_back({tok[i], parseValue(tok[i + 1])});
    blk.osdiDevices.push_back(o);
}

// T<name> <p1+> <p1-> <p2+> <p2-> Z0=<z0> (TD=<delay> | F=<f> [NL=<wavelengths, 0.25>]) [IC=v1,i1,v2,i2]   (ngspice TRA)
void SPICEParser::parseTLine(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 8);
    TransmissionLine t{};
    t.name = tok[0];
    t.nodePort1Pos = nodeIndex(tok[1]); t.nodePort1Neg = nodeIndex(tok[2]);
    t.nodePort2Pos = nodeIndex(tok[3]); t.nodePort2Neg = nodeIndex(tok[4]);
    double td = 0.0, f = 0.0, nl = 0.25;
    for (size_t i = 5; i + 1 < tok.size(); ++i) {
        const std::string k = toUpper(tok[i]);
        if (k == "Z0" || k == "ZO") t.characteristicImpedance_Z0_ohms = parseValue(tok[++i]);
        else if (k == "TD") td = parseValue(tok[++i]);
        else if (k == "F") f = parseValue(tok[++i]);
        else if (k == "NL") nl = parseValue(tok[++i]);
        else if (k == "IC" && i + 4 < tok.size()) {
            for (double& v : t.ic) v = parseValue(tok[++i]);
            t.icGiven = true;
        } else throw std::runtime_error(tok[0] + ": parameter " + k + " is not supported yet");
    }
    t.propagationDelay_Td_s = td > 0.0 ? td : f > 0.0 ? nl / f : 0.0;
    if (!(t.characteristicImpedance_Z0_ohms > 0.0)) throw std::runtime_error(tok[0] + ": Z0 must be given and > 0");
    if (!(t.propagationDelay_Td_s > 0.0)) throw std::runtime_error(tok[0] + ": TD (or F and NL) must be given and > 0");
    blk.transmissionLines.push_back(t);
}

// E<name> <n+> <n-> <nc+> <nc-> <gain>
// E/G/F/H: only the linear form. POLY / TABLE / VALUE / VOL / CUR / LAPLACE ...
// used to be read as node names (E2 2 0 VOL=2*V(1) -> nodes "VOL", "2*V").
void SPICEParser::requireLinearForm(const std::vector<std::string>& tok, size_t n) {
    for (size_t i = 3; i < tok.size(); ++i) {
        const std::string k = toUpper(tok[i]);
        if (k == "POLY" || k == "TABLE" || k == "VALUE" || k == "VOL" || k == "CUR" || k == "LAPLACE" ||
            k == "FREQ" || k == "CHEBYSHEV")
            throw std::runtime_error(tok[0] + ": " + k + " form is not supported yet");
    }
    if (tok.size() != n)
        throw std::runtime_error(tok[0] + ": expected " + std::to_string(n - 1) + " fields (linear form)");
}

void SPICEParser::parseVCVS(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireLinearForm(tok, 6);
    VCVS e;
    e.outPos  = nodeIndex(tok[1]);
    e.outNeg  = nodeIndex(tok[2]);
    e.ctrlPos = nodeIndex(tok[3]);
    e.ctrlNeg = nodeIndex(tok[4]);
    e.gain    = parseValue(tok[5]);
    e.name    = toUpper(tok[0]);
    blk.voltageControlledVoltageSources.push_back(e);
}

// G<name> <n+> <n-> <nc+> <nc-> <gm>
void SPICEParser::parseVCCS(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireLinearForm(tok, 6);
    VCCS g;
    g.outPos  = nodeIndex(tok[1]);
    g.outNeg  = nodeIndex(tok[2]);
    g.ctrlPos = nodeIndex(tok[3]);
    g.ctrlNeg = nodeIndex(tok[4]);
    g.gm      = parseValue(tok[5]) * mult_;
    blk.voltageControlledCurrentSources.push_back(g);
}

// ============================================================================
// Directive Parsers
// ============================================================================

// .MODEL <name> <type> ([param=val]...)
void SPICEParser::parseModelCard(const std::vector<std::string>& tok) {
    if (tok.size() < 3) return;
    ModelCard card;
    card.name = tok[1];
    card.type = toUpper(tok[2]);
    std::vector<std::string> t(tok.begin(), tok.begin() + 3);
    if (card.type.rfind("VDMOS", 0) == 0) {
        // ngspice inp_vdmos_model: PCHAN (or VDMOSP) makes it p-channel; the datasheet notes RON, VDS, QG and MFG
        // (MFG a name) are dropped
        bool p = card.type == "VDMOSP";
        for (size_t i = 3; i < tok.size(); ++i) {
            const std::string k = toUpper(tok[i]);
            if (k == "PCHAN" || k == "NCHAN") { p = p || k == "PCHAN"; continue; }
            if ((k == "RON" || k == "VDS" || k == "QG" || k == "MFG") && i + 1 < tok.size()) { ++i; continue; }
            t.push_back(tok[i]);
        }
        card.type = p ? "VDMOSP" : "VDMOSN";
    } else t = tok;

    // Vector parameters, KEY=[v1 v2 ...] (XSPICE code models): collected and taken out of the key=value pairs
    {
        std::vector<std::string> u(t.begin(), t.begin() + std::min<size_t>(3, t.size()));
        for (size_t i = 3; i < t.size(); ++i) {
            if (i + 1 >= t.size() || t[i + 1][0] != '[') { u.push_back(t[i]); continue; }
            const std::string key = toUpper(t[i]);
            std::vector<double> v;
            std::string w = t[++i].substr(1);
            for (;;) {
                const bool last = !w.empty() && w.back() == ']';
                if (last) w.pop_back();
                if (!w.empty()) v.push_back(parseValue(w));
                if (last) break;
                if (++i >= t.size()) throw std::runtime_error("model " + card.name + ": " + key + "=[ without ]");
                w = t[i];
            }
            card.vectors[key] = v;
        }
        t = u;
    }

    // Parse optional parameters (key=value pairs flattened by tokenizer)
    // LTRA's flags are bare words (nosteplimit, lininterp, ...): each becomes key = 1
    if (card.type == "LTRA") {
        static const std::set<std::string> flags = {"LTRA", "NOCONTROL", "STEPLIMIT", "NOSTEPLIMIT", "LININTERP", "QUADINTERP",
                                                    "MIXEDINTERP", "TRUNCNR", "TRUNCDONTCUT"};
        std::vector<std::string> u(t.begin(), t.begin() + std::min<size_t>(3, t.size()));
        for (size_t i = 3; i < t.size(); ++i) {
            const std::string k = toUpper(t[i]);
            if (flags.count(k) && (i + 1 >= t.size() || flags.count(toUpper(t[i + 1])) || !isValue(t[i + 1]))) { card.params[k] = 1.0; continue; }
            u.push_back(t[i]);
        }
        t = u;
    }
    // the code models' string parameters (quotes optional, as ngspice)
    static const std::map<std::string, std::string> stringKeys = {
        {"D_LUT", "TABLE_VALUES"}, {"D_SOURCE", "INPUT_FILE"}, {"D_STATE", "STATE_FILE"}, {"MULTI_INPUT_PWL", "MODEL"}, {"FILESOURCE", "FILE"}, {"XFER", "FILE"}};
    for (size_t i = 3; i + 1 < t.size(); i += 2) {
        std::string key = toUpper(t[i]);
        const std::string& raw = t[i + 1];
        if (const auto sk = stringKeys.find(card.type); sk != stringKeys.end() && sk->second == key) {
            card.strings[key] = raw.size() >= 2 && raw.front() == '"' && raw.back() == '"' ? raw.substr(1, raw.size() - 2) : raw;
            continue;
        }
        const std::string word = toUpper(t[i + 1]);  // XSPICE booleans (inertial_delay=false)
        double val = word == "TRUE" ? 1.0 : word == "FALSE" ? 0.0 : parseValue(t[i + 1]);
        card.params[key] = val;

        // Extract well-known fields
        if (key == "LEVEL") card.level = static_cast<int>(val);
    }

    if (!modelCards_.emplace(toUpper(card.name), card).second)  // ngspice warns and keeps the first
        throw std::runtime_error("model " + card.name + " defined twice");
}

// .SUBCKT <name> <port1> <port2> ...
void SPICEParser::parseSubcktBegin(const std::vector<std::string>& tok) {
    requireFields(tok, 2);
    if (inSubckt_)
        throw std::runtime_error("nested .subckt definitions are not supported (missing .ends for " +
                                 subcktName_ + "?)");
    inSubckt_ = true;
    subcktName_ = toUpper(tok[1]);
    if (result_.netlist.blockDefinitions.count(subcktName_))
        throw std::runtime_error("subcircuit " + subcktName_ + " defined twice");
    subcktBlock_ = TensorBlock{};
    subcktBlock_.name = subcktName_;
    subcktNodeMap_.clear();
    nextLocalNode_ = 1;
    savedInductorIndex_.swap(inductorIndex_);  // K cards inside see only this body's L
    inductorIndex_.clear();

    subcktPorts_.clear();
    for (size_t i = 2; i < tok.size(); ++i) {
        subcktPorts_.push_back(tok[i]);
        subcktBlock_.pinNames.push_back(tok[i]);
        noAnalogRef_ = true;
        subcktBlock_.portNodeIndices.push_back(nodeIndex(tok[i]));  // local 1..P
        noAnalogRef_ = false;
    }
}

// H<name> <n+> <n-> <Vctrl> <transresistance>
void SPICEParser::parseCCVS(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireLinearForm(tok, 5);
    CCVS h{};
    h.outPos = nodeIndex(tok[1]);
    h.outNeg = nodeIndex(tok[2]);
    h.ctrlPos = h.ctrlNeg = 0;
    h.ctrlName = toUpper(tok[3]);
    h.rm = parseValue(tok[4]);
    h.name = toUpper(tok[0]);
    blk.currentControlledVoltageSources.push_back(h);
}

// F<name> <n+> <n-> <Vctrl> <gain>
void SPICEParser::parseCCCS(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireLinearForm(tok, 5);
    CCCS f{};
    f.outPos = nodeIndex(tok[1]);
    f.outNeg = nodeIndex(tok[2]);
    f.ctrlPos = f.ctrlNeg = 0;
    f.ctrlName = toUpper(tok[3]);
    f.alpha = parseValue(tok[4]) * mult_;
    blk.currentControlledCurrentSources.push_back(f);
}

bool SPICEParser::nonlinearForm(const std::vector<std::string>& tok) {
    if (tok.size() < 4) return false;
    const std::string k = toUpper(tok[3]);
    return k == "POLY" || k == "VALUE" || k == "VOL" || k == "CUR" || k == "TABLE";
}

static std::string fmt(double v) {
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

// B<name> n+ n- V=expr | I=expr;  E/G<name> n+ n- VALUE={expr} | TABLE {expr} = (x,y) ... | POLY(n) ...;
// F/H<name> n+ n- POLY(n) Vsrc ... coefficients. `line` is the whole card (numparam::expand left the
// expression as canonical text): tokenize() would flatten its punctuation.
void SPICEParser::parseBehav(const std::string& line, const std::vector<std::string>& tok, TensorBlock& blk) {
    const char t = static_cast<char>(std::toupper(static_cast<unsigned char>(tok[0][0])));
    requireFields(tok, 4);
    Behavioural b;
    b.name = toUpper(tok[0]);
    b.outPos = nodeIndex(tok[1]);
    b.outNeg = nodeIndex(tok[2]);
    b.voltage = t == 'E' || t == 'H';
    const std::string kw = toUpper(tok[3]);
    std::istringstream ws(line);
    std::vector<std::string> w;
    for (std::string x; ws >> x;) w.push_back(x);
    auto afterEq = [&] {
        const size_t eq = line.find('=');
        return eq == std::string::npos ? std::string() : line.substr(eq + 1);
    };
    std::string text;
    if (t == 'B') {
        if (kw != "V" && kw != "I") throw std::runtime_error(tok[0] + ": expected V=<expression> or I=<expression>");
        b.voltage = kw == "V";
        text = afterEq();
    } else if (kw == "POLY") {
        // SPICE3: p0 + sum p_k m_k, the monomials m_k of the controls x1..xn by degree, each degree in
        // non-decreasing index order (x1^2, x1*x2, x2^2, ...); fewer coefficients than terms leave the rest 0
        if (tok.size() < 5) throw std::runtime_error(tok[0] + ": POLY needs its dimension");
        const int n = static_cast<int>(parseValue(tok[4]));
        const bool ctlNodes = t == 'E' || t == 'G';
        const size_t first = 5 + (ctlNodes ? 2 : 1) * static_cast<size_t>(std::max(n, 0));
        if (n < 1 || tok.size() < first) throw std::runtime_error(tok[0] + ": POLY(n) needs n >= 1 and its controls");
        std::vector<std::string> x;
        for (int i = 0; i < n; ++i)
            x.push_back(ctlNodes ? "V(" + tok[5 + 2 * i] + "," + tok[6 + 2 * i] + ")" : "I(" + tok[5 + i] + ")");
        std::vector<double> c;
        for (size_t i = first; i < tok.size(); ++i) {
            if (toUpper(tok[i]) == "IC") throw std::runtime_error(tok[0] + ": POLY IC is not supported yet");
            c.push_back(parseValue(tok[i]));
        }
        text = "0";
        std::vector<std::vector<int>> mono = {{}};  // the monomial of coefficient k, as control indices
        for (int d = 1; mono.size() < c.size(); ++d) {
            std::vector<std::vector<int>> deg = {{}};
            for (int pos = 0; pos < d; ++pos) {
                std::vector<std::vector<int>> next;
                for (const auto& m : deg)
                    for (int i = m.empty() ? 0 : m.back(); i < n; ++i) { next.push_back(m); next.back().push_back(i); }
                deg = std::move(next);
            }
            mono.insert(mono.end(), deg.begin(), deg.end());
        }
        for (size_t k = 0; k < c.size(); ++k) {
            text += "+(" + fmt(c[k]) + ")";
            for (int i : mono[k]) text += "*" + x[i];
        }
    } else if (kw == "TABLE") {
        if (w.size() < 7 || w[5] != "=") throw std::runtime_error(tok[0] + ": expected TABLE {expression} = (x,y) ...");
        std::string pts = line.substr(line.find(" = ") + 3);
        for (auto& ch : pts) if (ch == '(' || ch == ')' || ch == ',') ch = ' ';
        std::istringstream in(pts);
        std::vector<std::string> v;
        for (std::string x; in >> x;) v.push_back(x);
        if (v.size() < 4 || v.size() % 2) throw std::runtime_error(tok[0] + ": TABLE needs at least two (x,y) points");
        text = "table(" + w[4];
        double prev = -INFINITY;
        for (size_t i = 0; i < v.size(); i += 2) {
            const double xv = parseValue(v[i]);
            if (!(xv > prev)) throw std::runtime_error(tok[0] + ": TABLE x values must increase");
            prev = xv;
            text += "," + fmt(xv) + "," + fmt(parseValue(v[i + 1]));
        }
        text += ")";
    } else if (kw == "VALUE" || kw == "VOL" || kw == "CUR") {
        if (t == 'F' || t == 'H') throw std::runtime_error(tok[0] + ": " + kw + " form is not supported yet");
        text = afterEq();
    } else throw std::runtime_error(tok[0] + ": " + kw + " form is not supported yet");

    numparam::Expr e = numparam::parse(text, [](const std::string&, double&) { return false; });
    for (const auto& r : e.refs) {
        Behavioural::Ref br;
        if (r.kind == 'V') { br.a = nodeIndex(r.a); br.b = r.b.empty() ? 0 : nodeIndex(r.b); }
        else { br.kind = -1; br.src = r.a; }
        b.refs.push_back(br);
    }
    b.expr = std::make_shared<const numparam::Expr>(std::move(e));
    blk.behavioural.push_back(std::move(b));
}

void SPICEParser::resolveControlledSources(TensorBlock& blk) {
    std::map<std::string, int> bvRank;  // voltage behavioural sources, in branch order
    for (const auto& b : blk.behavioural) if (b.voltage) bvRank.emplace(b.name, (int)bvRank.size());
    for (auto& b : blk.behavioural)
        for (auto& r : b.refs) {
            if (r.kind != -1) continue;
            auto found = [&](const auto& list, int kind) {
                for (size_t i = 0; i < list.size(); ++i)
                    if (toUpper(list[i].name) == r.src) { r.kind = kind; r.a = (int)i; return true; }
                return false;
            };
            if (!found(blk.voltageSources, 1) && !found(blk.inductors, 2)) {
                if (auto it = bvRank.find(r.src); it != bvRank.end()) { r.kind = 3; r.a = it->second; }
                else throw std::runtime_error(b.name + ": I(" + r.src + "): no V source, inductor or B voltage source of that name" +
                                              (blk.name.empty() ? "" : " in subcircuit " + blk.name));
            }
        }
    auto find = [&](const std::string& name) {
        for (size_t i = 0; i < blk.voltageSources.size(); ++i)
            if (blk.voltageSources[i].name == name) return (int)i;
        throw std::runtime_error("controlling source " + name + " not found" +
                                 (blk.name.empty() ? "" : " in subcircuit " + blk.name) +
                                 " (F/H must reference a V source in the same block)");
    };
    for (auto& h : blk.currentControlledVoltageSources) h.ctrlSource = find(h.ctrlName);
    for (auto& f : blk.currentControlledCurrentSources) f.ctrlSource = find(f.ctrlName);
    for (auto& s : blk.switches) if (!s.ctrlName.empty()) s.ctrlSource = find(s.ctrlName);
}

void SPICEParser::parseSubcktEnd() {
    if (!inSubckt_) throw std::runtime_error(".ends without .subckt");
    resolveControlledSources(subcktBlock_);
    inSubckt_ = false;
    subcktNames_[subcktName_] = subcktNodeMap_;
    inductorIndex_.swap(savedInductorIndex_);
    subcktBlock_.numInternalNodes = nextLocalNode_ - 1;  // ports + internal nodes
    for (const auto& ic : subcktICs_[subcktName_])
        if (!subcktNodeMap_.count(ic.node))
            throw std::runtime_error(std::string(ic.nodeset ? ".nodeset" : ".ic") + " (line " + std::to_string(ic.line) +
                                     ") names unknown node '" + ic.node + "' in subcircuit " + subcktName_);

    result_.netlist.defineBlock(subcktBlock_);
    result_.subcircuitsCreated++;
}

// One instance of `pi.subckt` with ports on global nodes `portNodes`: fresh
// global nodes for its internal nodes, then its own X cards, recursively.
void SPICEParser::expandInstance(const PendingInstance& pi, const std::vector<int>& portNodes,
                                 int depth, const std::string& path, const std::vector<std::string>& portNames, int topLine) {
    if (depth > 64) throw std::runtime_error("subcircuit nesting deeper than 64 (recursive .subckt?)");
    auto it = result_.netlist.blockDefinitions.find(pi.subckt);
    if (it == result_.netlist.blockDefinitions.end())
        throw std::runtime_error("unknown subcircuit '" + pi.subckt + "'");
    const TensorBlock& blk = it->second;
    if (portNodes.size() != blk.pinNames.size())
        throw std::runtime_error(pi.subckt + " has " + std::to_string(blk.pinNames.size()) +
                                 " pins but the instance connects " + std::to_string(portNodes.size()));

    std::vector<int> map(portNodes);  // local node l -> global map[l-1]
    const auto& globals = subcktGlobals_[pi.subckt];  // internal locals named in .global
    for (int l = (int)map.size() + 1; l <= blk.numInternalNodes; ++l) {
        auto g = globals.find(l);
        if (g == globals.end()) { map.push_back(nextNode_++); continue; }
        auto [node, isNew] = nodeMap_.emplace(g->second, nextNode_);
        if (isNew) ++nextNode_;
        map.push_back(node->second);
    }
    const std::string full = path + pi.name;
    result_.netlist.addInstance(pi.subckt, map, full);
    for (const auto& [local, l] : subcktNames_[pi.subckt])  // internal nodes are addressable as x1.n
        if (l > (int)portNodes.size()) result_.nodeIndex.emplace(full + "." + local, map[l - 1]);
    for (const auto& ic : subcktICs_[pi.subckt])  // a body's .ic / .nodeset applies to each instance
        (ic.nodeset ? result_.netlist.nodeSets : result_.netlist.nodeICs).push_back({map[subcktNames_[pi.subckt].at(ic.node) - 1], ic.value});

    for (int l : analogLocal_[pi.subckt]) if (l >= 1 && l <= (int)map.size()) analogGlobal_.insert(map[l - 1]);
    // the flattened name of a body's node: a pin's is the parent's net, ground and .global ones stay, others x1.n
    auto flat = [&](const std::string& local) {
        const std::string u = toUpper(local);
        if (u == "0" || u == "GND" || globals_.count(u)) return u;
        for (size_t k = 0; k < blk.pinNames.size(); ++k)
            if (toUpper(blk.pinNames[k]) == u && k < portNames.size()) return portNames[k];
        return full + "." + u;
    };
    for (const auto& a : subcktA_[pi.subckt]) {  // the body's XSPICE devices, on the flattened nets
        PendingA x{full + "." + toUpper(a.name), a.model, {}, topLine, {}};
        for (const auto& w : a.raw) {
            if (w.empty() || w[0] == '%' || w == "[" || w == "]") { x.raw.push_back(w); continue; }
            const bool inv = w[0] == '~';
            const std::string base = inv ? w.substr(1) : w;
            x.raw.push_back(toUpper(base) == "NULL" ? base : (inv ? "~" : "") + flat(base));
        }
        for (const auto& g : a.groups) {
            std::vector<std::string> fg;
            for (const auto& n : g) {
                const bool inv = !n.empty() && n[0] == '~';
                const std::string base = inv ? n.substr(1) : n;
                fg.push_back(toUpper(base) == "NULL" ? base : (inv ? "~" : "") + flat(base));
            }
            x.groups.push_back(fg);
        }
        pendingA_.push_back(x);
    }
    for (const auto& child : subcktChildren_[pi.subckt]) {
        std::vector<int> childPorts;
        std::vector<std::string> childNames;
        for (int l : child.nodes) childPorts.push_back(l == 0 ? 0 : map[l - 1]);
        for (const auto& n : child.portNames) childNames.push_back(flat(n));
        expandInstance(child, childPorts, depth + 1, full + ".", childNames, topLine);
    }
}

// .PARAM <name>=<value> [<name>=<value>]... (numparam::expand's evaluated globals)
void SPICEParser::parseParam(const std::vector<std::string>& tok) {
    for (size_t i = 1; i + 1 < tok.size(); i += 2) params_[toUpper(tok[i])] = parseValue(tok[i + 1]);
}

} // namespace acutesim

namespace acutesim {

// A<name> <ports> <model>: a port is a node, NULL, ~node (an inverted digital input), or a [ ... ] vector;
// %d / %v before a port name its type (the model decides it anyway).
void SPICEParser::parseXspice(const std::string& line, int lineNum) {
    std::string spaced;
    for (char c : line)  // commas separate as blanks do (ngspice INPgetNetTok): %vd(a, b), %vd(a,b)
        spaced += c == '[' || c == ']' ? std::string(" ") + c + " " : c == '(' || c == ')' || c == ',' ? std::string(" ") : std::string(1, c);
    std::istringstream ss(spaced);
    std::vector<std::string> w;
    for (std::string t; ss >> t;) w.push_back(t);
    if (w.size() < 3) throw std::runtime_error(w[0] + ": expected ports, then a model name");
    PendingA a{w[0], w.back(), {}, lineNum, std::vector<std::string>(w.begin() + 1, w.end() - 1)};
    for (size_t i = 1; i + 1 < w.size(); ++i) {
        if (w[i][0] == '%') continue;  // a port type: the code model checks it (resolveXspice)
        if (w[i] == "]") throw std::runtime_error(a.name + ": ']' without '['");
        if (w[i] != "[") { a.groups.push_back({w[i]}); continue; }
        std::vector<std::string> g;
        for (++i; i + 1 < w.size() && w[i] != "]"; ++i)
            if (w[i][0] != '%') g.push_back(w[i]);
        if (i + 1 >= w.size()) throw std::runtime_error(a.name + ": '[' without ']'");
        a.groups.push_back(g);
    }
    (inSubckt_ ? subcktA_[subcktName_] : pendingA_).push_back(a);
}

// The A devices against their .model cards: ngspice 47's d_buffer ... d_dff, adc_bridge and dac_bridge
// (parameters and defaults probed; see physics/xspice.h). A dac_bridge output becomes a voltage source.
std::string SPICEParser::dataPath(const std::string& file) const {
    const std::filesystem::path path(file);
    if (path.is_relative() && !baseDir_.empty() && std::filesystem::is_regular_file(std::filesystem::path(baseDir_) / path))
        return (std::filesystem::path(baseDir_) / path).string();
    return file;
}

// ngspice d_source's vector file: rows "time state..." (one 12-state token per output, as 0s 1r Uz), lines starting
// with * and blank lines skipped, tokens split at blanks = ( ) and commas. ngspice reports a bad file and holds the
// outputs at U; here it is an error.
void SPICEParser::readDigitalSource(::xspice::Inst& x, const std::string& file, const std::string& model) {
    std::ifstream in(dataPath(file));
    if (!in) throw std::runtime_error("model " + model + ": cannot open d_source input_file " + file);
    static const std::string states[12] = {"0s", "1s", "Us", "0r", "1r", "Ur", "0z", "1z", "Uz", "0u", "1u", "Uu"};
    std::string line;
    for (int n = 1; std::getline(in, line); ++n) {
        if (line.empty() || line[0] == '*') continue;
        for (char& ch : line) if (ch == '=' || ch == '(' || ch == ')' || ch == ',') ch = ' ';
        std::istringstream ss(line);
        std::vector<std::string> tok;
        for (std::string w; ss >> w;) tok.push_back(w);
        if (tok.empty()) continue;
        const std::string where = "model " + model + ": " + file + " line " + std::to_string(n) + ": ";
        if (tok.size() != x.out.size() + 1)
            throw std::runtime_error(where + "expected a time and " + std::to_string(x.out.size()) + " states");
        const double t = parseValue(tok[0]);
        if (!x.times.empty() && !(t > x.times.back())) throw std::runtime_error(where + "times must increase");
        // ngspice queues each row 0.1 ns early and posts it with a 0.1 ns delay: a row in (0, 0.1 ns) lands before t = 0
        if (t < 0.0 || (t > 0.0 && t < 1e-10)) throw std::runtime_error(where + "a time in (0, 0.1 ns) or below 0 is not supported");
        std::vector<::xspice::Val> row;
        for (size_t k = 1; k < tok.size(); ++k) {
            const auto* hit = std::find(states, states + 12, tok[k]);
            if (hit == states + 12) throw std::runtime_error(where + "unknown state " + tok[k]);
            const int i = static_cast<int>(hit - states);
            row.push_back({static_cast<::xspice::Lv>(i % 3), static_cast<::xspice::St>(i / 3)});
        }
        x.times.push_back(t);
        x.rows.push_back(std::move(row));
    }
}

// ngspice d_state's state table: a state's first row "state out... in... -> next" (12-state outputs, inputs 0 1 x),
// then rows "in... -> next" for the same state; tokens and comments as d_source. ngspice falls back to row 0 for a
// state it cannot find and stops at rows of one state split apart; both are errors here, as is a bad file.
void SPICEParser::readDigitalState(::xspice::Inst& x, const std::string& file, const std::string& model) {
    std::ifstream in(dataPath(file));
    if (!in) throw std::runtime_error("model " + model + ": cannot open d_state state_file " + file);
    static const std::string states[12] = {"0s", "1s", "Us", "0r", "1r", "Ur", "0z", "1z", "Uz", "0u", "1u", "Uu"};
    const size_t ni = x.in.size() - 2, no = x.out.size();
    auto integer = [&](const std::string& w, const std::string& where) {
        const double v = parseValue(w);
        if (v != std::floor(v) || std::abs(v) > 1e9) throw std::runtime_error(where + "state " + w + " is not an integer");
        return static_cast<int>(v);
    };
    std::string line;
    for (int n = 1; std::getline(in, line); ++n) {
        if (line.empty() || line[0] == '*') continue;
        for (char& ch : line) if (ch == '=' || ch == '(' || ch == ')' || ch == ',') ch = ' ';
        std::istringstream ss(line);
        std::vector<std::string> tok;
        for (std::string w; ss >> w;) tok.push_back(w);
        if (tok.empty()) continue;
        const std::string where = "model " + model + ": " + file + " line " + std::to_string(n) + ": ";
        const bool header = tok.size() == ni + no + 3;
        if (!header && tok.size() != ni + 2)
            throw std::runtime_error(where + "expected state, " + std::to_string(no) + " outputs, " + std::to_string(ni) +
                                     " inputs, -> and the next state (or only the inputs, -> and the next state)");
        if (!header && x.state.empty()) throw std::runtime_error(where + "the first row needs its state and outputs");
        size_t k = 0;
        if (header) {
            x.state.push_back(integer(tok[k++], where));
            std::vector<::xspice::Val> row;
            for (size_t o = 0; o < no; ++o, ++k) {
                const auto* hit = std::find(states, states + 12, tok[k]);
                if (hit == states + 12) throw std::runtime_error(where + "unknown output state " + tok[k]);
                const int i = static_cast<int>(hit - states);
                row.push_back({static_cast<::xspice::Lv>(i % 3), static_cast<::xspice::St>(i / 3)});
            }
            x.rows.push_back(std::move(row));
        } else {
            x.state.push_back(x.state.back());
            x.rows.push_back(x.rows.back());
        }
        std::vector<::xspice::Lv> m;
        for (size_t j = 0; j < ni; ++j, ++k) {
            const std::string& w = tok[k];
            if (w != "0" && w != "1" && w != "x" && w != "X") throw std::runtime_error(where + "unknown input value " + w);
            m.push_back(w == "0" ? ::xspice::L0 : w == "1" ? ::xspice::L1 : ::xspice::LU);
        }
        x.match.push_back(std::move(m));
        if (tok[k++] != "->") throw std::runtime_error(where + "expected -> before the next state");
        x.nextState.push_back(integer(tok[k], where));
    }
    if (x.state.empty()) throw std::runtime_error("model " + model + ": d_state state_file " + file + " has no states");
    // ngspice keeps only a state's contiguous rows and takes its outputs from the first
    for (size_t r = 1; r < x.state.size(); ++r)
        if (x.state[r] != x.state[r - 1] && std::find(x.state.begin(), x.state.begin() + r, x.state[r]) != x.state.begin() + r)
            throw std::runtime_error("model " + model + ": d_state state " + std::to_string(x.state[r]) + "'s rows are not contiguous");
    auto known = [&](int st) { return std::find(x.state.begin(), x.state.end(), st) != x.state.end(); };
    if (!known(x.icount)) throw std::runtime_error("model " + model + ": reset_state " + std::to_string(x.icount) + " is not in " + file);
    for (int st : x.nextState)
        if (!known(st)) throw std::runtime_error("model " + model + ": next state " + std::to_string(st) + " is not in " + file);
}

void SPICEParser::resolveXspice(int& at) {
    if (pendingA_.empty()) return;
    using xspice::Kind;
    static const std::map<std::string, Kind> kinds = {
        {"D_BUFFER", Kind::BUF}, {"D_INVERTER", Kind::INV}, {"D_AND", Kind::AND}, {"D_NAND", Kind::NAND},
        {"D_OR", Kind::OR}, {"D_NOR", Kind::NOR}, {"D_XOR", Kind::XOR}, {"D_XNOR", Kind::XNOR},
        {"D_TRISTATE", Kind::TRI}, {"D_PULLUP", Kind::PULLUP}, {"D_PULLDOWN", Kind::PULLDOWN}, {"D_DFF", Kind::DFF},
        {"D_JKFF", Kind::JKFF}, {"D_TFF", Kind::TFF}, {"D_SRFF", Kind::SRFF},
        {"D_DLATCH", Kind::DLATCH}, {"D_SRLATCH", Kind::SRLATCH}, {"D_OPEN_C", Kind::OPENC}, {"D_OPEN_E", Kind::OPENE},
        {"D_FDIV", Kind::FDIV}, {"D_LUT", Kind::LUT}, {"D_SOURCE", Kind::SOURCE}, {"D_OSC", Kind::OSC}, {"D_PWM", Kind::PWM}, {"D_STATE", Kind::STATE}, {"D_RAM", Kind::RAM},
        {"ADC_BRIDGE", Kind::ADC}, {"DAC_BRIDGE", Kind::DAC}};
    auto net = std::make_shared<xspice::Net>();
    std::map<std::string, int> dnode;  // upper-case name -> digital node
    std::set<int> orphan;              // node indices X lines gave digital nets
    auto analogNode = [&](const std::string& n, const std::string& inst) {
        const std::string key = toUpper(n);
        if (const auto a = result_.nodeIndex.find(key); a != result_.nodeIndex.end()) {
            analogTop_.insert(a->second);
            return a->second;
        }
        if (key.find('.') != std::string::npos)
            throw std::runtime_error(inst + ": analog port " + n + " is on no analog element of its subcircuit");
        const int idx = nodeIndex(n);  // a top-level node only this bridge names
        result_.nodeIndex[key] = idx;
        result_.netlist.numGlobalNodes = std::max(result_.netlist.numGlobalNodes, idx);
        return idx;
    };
    auto digital = [&](const std::string& n, const std::string& inst, bool nullOk) -> xspice::Port {
        if (toUpper(n) == "NULL") {
            if (!nullOk) throw std::runtime_error(inst + ": this port cannot be NULL");
            return {};
        }
        const bool inv = n[0] == '~';  // inverted: an input reads, an output drives, the inverse
        const std::string name = inv ? n.substr(1) : n, key = toUpper(name);
        if (const auto a = result_.nodeIndex.find(key); a != result_.nodeIndex.end()) {
            if (analogTop_.count(a->second) || analogGlobal_.count(a->second))
                throw std::runtime_error(inst + ": node " + name + " is analog; a digital port needs an event node");
            orphan.insert(a->second);  // only X ports and pins name it: no analog node
        }
        auto [it, fresh] = dnode.emplace(key, (int)net->nodes.size());
        if (fresh) {
            std::string low = name;
            for (auto& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            net->nodes.push_back(low);
        }
        return {it->second, inv};
    };
    for (const auto& a : pendingA_) {
        at = a.line;
        const auto mc = modelCards_.find(toUpper(a.model));
        if (mc == modelCards_.end()) throw std::runtime_error(a.name + ": no .model card " + a.model);
        const ModelCard* c = &mc->second;
        static const std::set<std::string> analogKinds = {"GAIN", "SUMMER", "MULT", "DIVIDE", "LIMIT", "CLIMIT", "PWL", "ASWITCH", "INT", "D_DT", "S_XFER", "SLEW", "HYST", "SINE", "TRIANGLE", "SQUARE", "ZENER", "MULTI_INPUT_PWL", "OTA", "ILIMIT", "ONESHOT", "POTENTIOMETER", "CAPACITORIC", "INDUCTORIC", "SIDIODE", "PSWITCH", "MEMRISTOR", "FILESOURCE", "PWLTS", "XFER", "CORE", "LCOUPLE"};
        if (analogKinds.count(c->type)) { resolveXspiceAnalog(a, *c, analogNode); continue; }
        const auto k = kinds.find(c->type);
        if (k == kinds.end()) throw std::runtime_error(a.name + ": XSPICE model type " + c->type + " is not supported yet");
        for (const auto& w : a.raw)
            if (w[0] == '%' && w != "%d" && w != "%v") throw std::runtime_error(a.name + ": port type " + w + " is not supported yet");
        for (const auto& [key, v] : c->vectors)
            if (!(k->second == Kind::OSC && (key == "CNTL_ARRAY" || key == "FREQ_ARRAY")) &&
                !(k->second == Kind::PWM && (key == "CNTL_ARRAY" || key == "DC_ARRAY")))
                throw std::runtime_error("model " + c->name + ": parameter " + key + " is not a " + c->type + " parameter");
        xspice::Inst x;
        x.name = a.name;
        x.kind = k->second;
        auto single = [&](size_t g) {
            if (g >= a.groups.size() || a.groups[g].size() != 1) throw std::runtime_error(a.name + ": expected a single port at position " + std::to_string(g + 1));
            return a.groups[g][0];
        };
        auto need = [&](size_t n) {
            if (a.groups.size() != n) throw std::runtime_error(a.name + ": " + c->type + " takes " + std::to_string(n) + " ports, got " + std::to_string(a.groups.size()));
        };
        std::set<std::string> allowed;
        switch (x.kind) {
        case Kind::BUF: case Kind::INV:
            need(2);
            x.in = {digital(single(0), a.name, false)};
            x.out = {digital(single(1), a.name, false)};
            allowed = {"RISE_DELAY", "FALL_DELAY", "INPUT_LOAD", "INERTIAL_DELAY"};
            break;
        case Kind::OPENC: case Kind::OPENE:
            need(2);
            x.in = {digital(single(0), a.name, false)};
            x.out = {digital(single(1), a.name, false)};
            allowed = {x.kind == Kind::OPENC ? "FALL_DELAY" : "RISE_DELAY", "OPEN_DELAY", "INPUT_LOAD", "INERTIAL_DELAY"};
            break;
        case Kind::FDIV: {
            need(2);
            x.in = {digital(single(0), a.name, false)};
            x.out = {digital(single(1), a.name, false)};
            allowed = {"DIV_FACTOR", "HIGH_CYCLES", "I_COUNT", "RISE_DELAY", "FALL_DELAY", "FREQ_IN_LOAD"};
            auto integer = [&](const char* k, int dflt) {
                const double v = c->get(k, dflt);
                if (v != std::floor(v) || std::abs(v) > 1e9) throw std::runtime_error("model " + c->name + ": " + k + " must be an integer");
                return static_cast<int>(v);
            };
            x.div = integer("DIV_FACTOR", 2); x.high = integer("HIGH_CYCLES", 1); x.icount = integer("I_COUNT", 0);
            break;
        }
        case Kind::LUT: {
            need(2);
            for (const auto& n : a.groups[0]) x.in.push_back(digital(n, a.name, false));
            if (x.in.empty() || x.in.size() > 20) throw std::runtime_error(a.name + ": d_lut takes 1 to 20 inputs");
            x.out = {digital(single(1), a.name, false)};
            allowed = {"RISE_DELAY", "FALL_DELAY", "INPUT_LOAD"};
            const auto tv = c->strings.find("TABLE_VALUES");
            if (tv == c->strings.end()) throw std::runtime_error("model " + c->name + ": d_lut needs table_values=\"...\" (quoted)");
            // ngspice: '1' / '0' per entry, anything else U; a short table is padded with U, a long one cut
            x.table.assign(size_t{1} << x.in.size(), xspice::LU);
            for (size_t k = 0; k < x.table.size() && k < tv->second.size(); ++k)
                x.table[k] = tv->second[k] == '1' ? xspice::L1 : tv->second[k] == '0' ? xspice::L0 : xspice::LU;
            break;
        }
        case Kind::OSC: case Kind::PWM: {
            const bool pwm = x.kind == Kind::PWM;
            need(2);
            const std::string cn = single(0);
            if (dnode.count(toUpper(cn))) throw std::runtime_error(a.name + ": node " + cn + " is digital; cntl_in is analog");
            x.analog = {analogNode(cn, a.name)};
            x.out = {digital(single(1), a.name, false)};
            // ngspice's d_osc and d_pwm never apply rise_delay / fall_delay
            allowed = {pwm ? "FREQUENCY" : "DUTY_CYCLE", "INIT_PHASE", "RISE_DELAY", "FALL_DELAY"};
            const char* tab = pwm ? "DC_ARRAY" : "FREQ_ARRAY";
            x.ctl = c->vectors.count("CNTL_ARRAY") ? c->vectors.at("CNTL_ARRAY") : pwm ? std::vector<double>{-1.0, 1.0} : std::vector<double>{0.0, 1.0};
            x.freq = c->vectors.count(tab) ? c->vectors.at(tab) : pwm ? std::vector<double>{0.0, 1.0} : std::vector<double>{1e6, 2e6};
            if (x.ctl.size() < 2 || x.ctl.size() != x.freq.size())
                throw std::runtime_error("model " + c->name + ": cntl_array and " + (pwm ? "dc_array" : "freq_array") + " need the same size, 2 or more");
            for (size_t i = 0; i < x.ctl.size(); ++i) {
                if (i && !(x.ctl[i] > x.ctl[i - 1])) throw std::runtime_error("model " + c->name + ": cntl_array must increase");
                if (pwm) {  // ngspice: dc_array is limited to [0, 1], and 0 / 1 entries become 0.01 / 0.99 (its default [0 1] too)
                    if (!(x.freq[i] >= 0.0 && x.freq[i] <= 1.0)) throw std::runtime_error("model " + c->name + ": dc_array values must be in [0, 1]");
                    x.freq[i] = x.freq[i] <= 0.0 ? 0.01 : x.freq[i] >= 1.0 ? 0.99 : x.freq[i];
                } else if (!(x.freq[i] > 0.0)) throw std::runtime_error("model " + c->name + ": freq_array values must be > 0");
            }
            x.period = 1.0 / c->get("FREQUENCY", 1e6);
            if (pwm && !(c->get("FREQUENCY", 1e6) >= 1e-6)) throw std::runtime_error("model " + c->name + ": frequency must be >= 1e-6");
            x.duty = c->get("DUTY_CYCLE", 0.5);
            x.phase = c->get("INIT_PHASE", 0.0);
            if (!(x.duty >= 1e-6 && x.duty <= 0.999999)) throw std::runtime_error("model " + c->name + ": duty_cycle must be in [1e-6, 0.999999]");
            if (!(x.phase >= -180.0 && x.phase <= 360.0)) throw std::runtime_error("model " + c->name + ": init_phase must be in [-180, 360]");
            break;
        }
        case Kind::RAM: {
            need(5);  // [data_in] [data_out] [address] write_en [select]
            for (const auto& n : a.groups[0]) x.in.push_back(digital(n, a.name, false));
            for (const auto& n : a.groups[1]) x.out.push_back(digital(n, a.name, false));
            if (x.in.empty() || x.in.size() != x.out.size()) throw std::runtime_error(a.name + ": data_in and data_out need the same width, 1 or more");
            for (const auto& n : a.groups[2]) x.in.push_back(digital(n, a.name, false));
            x.addr = static_cast<int>(a.groups[2].size());
            // ponytail: the RAM is held as one Lv per bit (ngspice packs 4 per byte): 2^20 words is the bound kept
            if (x.addr < 1 || x.addr > 20) throw std::runtime_error(a.name + ": d_ram takes 1 to 20 address lines");
            x.in.push_back(digital(single(3), a.name, false));
            if (a.groups[4].empty() || a.groups[4].size() > 16) throw std::runtime_error(a.name + ": d_ram takes 1 to 16 select lines");
            for (const auto& n : a.groups[4]) x.in.push_back(digital(n, a.name, false));
            allowed = {"SELECT_VALUE", "IC", "READ_DELAY", "DATA_LOAD", "ADDRESS_LOAD", "SELECT_LOAD", "ENABLE_LOAD"};
            const double sv = c->get("SELECT_VALUE", 1.0);
            if (sv != std::floor(sv) || sv < 0.0 || sv > 65535.0) throw std::runtime_error("model " + c->name + ": select_value must be an integer in [0, 65535]");
            x.select = static_cast<int>(sv);
            break;
        }
        case Kind::STATE: {
            need(4);  // [in...] clk reset [out...]
            for (const auto& n : a.groups[0]) if (toUpper(n) != "NULL") x.in.push_back(digital(n, a.name, false));
            x.in.push_back(digital(single(1), a.name, false));
            x.in.push_back(digital(single(2), a.name, true));
            for (const auto& n : a.groups[3]) x.out.push_back(digital(n, a.name, false));
            if (x.out.empty()) throw std::runtime_error(a.name + ": no outputs");
            allowed = {"CLK_DELAY", "RESET_DELAY", "RESET_STATE", "INPUT_LOAD", "CLK_LOAD", "RESET_LOAD"};
            const double rs = c->get("RESET_STATE", 0.0);
            if (rs != std::floor(rs) || std::abs(rs) > 1e9) throw std::runtime_error("model " + c->name + ": reset_state must be an integer");
            x.icount = static_cast<int>(rs);
            readDigitalState(x, c->strings.count("STATE_FILE") ? c->strings.at("STATE_FILE") : "state.txt", c->name);
            break;
        }
        case Kind::SOURCE: {
            need(1);
            for (const auto& n : a.groups[0]) x.out.push_back(digital(n, a.name, false));
            if (x.out.empty()) throw std::runtime_error(a.name + ": no outputs");
            allowed = {"INPUT_LOAD"};
            readDigitalSource(x, c->strings.count("INPUT_FILE") ? c->strings.at("INPUT_FILE") : "source.txt", c->name);
            break;
        }
        case Kind::AND: case Kind::NAND: case Kind::OR: case Kind::NOR: case Kind::XOR: case Kind::XNOR:
            need(2);
            for (const auto& n : a.groups[0]) x.in.push_back(digital(n, a.name, false));
            if (x.in.empty()) throw std::runtime_error(a.name + ": no inputs");
            x.out = {digital(single(1), a.name, false)};
            allowed = {"RISE_DELAY", "FALL_DELAY", "INPUT_LOAD", "INERTIAL_DELAY"};
            break;
        case Kind::TRI:
            need(3);
            x.in = {digital(single(0), a.name, false), digital(single(1), a.name, false)};
            x.out = {digital(single(2), a.name, false)};
            allowed = {"DELAY", "INPUT_LOAD", "ENABLE_LOAD", "INERTIAL_DELAY"};
            break;
        case Kind::PULLUP: case Kind::PULLDOWN:
            need(1);
            x.out = {digital(single(0), a.name, false)};
            allowed = {"LOAD"};
            break;
        case Kind::DFF: case Kind::JKFF: case Kind::TFF: case Kind::SRFF: case Kind::DLATCH: case Kind::SRLATCH: {
            // data inputs, then clk (latches: enable) set reset out Nout
            const size_t nd = x.kind == Kind::JKFF || x.kind == Kind::SRFF || x.kind == Kind::SRLATCH ? 2 : 1;
            need(nd + 5);
            for (size_t k = 0; k <= nd; ++k) x.in.push_back(digital(single(k), a.name, false));
            x.in.push_back(digital(single(nd + 1), a.name, true));
            x.in.push_back(digital(single(nd + 2), a.name, true));
            x.out = {digital(single(nd + 3), a.name, true), digital(single(nd + 4), a.name, true)};
            if (x.kind == Kind::DLATCH || x.kind == Kind::SRLATCH)
                allowed = {"ENABLE_DELAY", "SET_DELAY", "RESET_DELAY", "IC", "RISE_DELAY", "FALL_DELAY", "ENABLE_LOAD", "SET_LOAD", "RESET_LOAD",
                           x.kind == Kind::DLATCH ? "DATA_DELAY" : "SR_DELAY", x.kind == Kind::DLATCH ? "DATA_LOAD" : "SR_LOAD"};
            else
                allowed = {"CLK_DELAY", "SET_DELAY", "RESET_DELAY", "IC", "RISE_DELAY", "FALL_DELAY", "CLK_LOAD", "SET_LOAD", "RESET_LOAD",
                           x.kind == Kind::DFF ? "DATA_LOAD" : x.kind == Kind::JKFF ? "JK_LOAD" : x.kind == Kind::TFF ? "T_LOAD" : "SR_LOAD"};
            break;
        }
        case Kind::ADC: case Kind::DAC: {
            need(2);
            const bool adc = x.kind == Kind::ADC;
            const auto& an = a.groups[adc ? 0 : 1];
            const auto& dn = a.groups[adc ? 1 : 0];
            if (an.size() != dn.size() || an.empty()) throw std::runtime_error(a.name + ": the analog and digital port vectors differ in size");
            for (size_t i = 0; i < an.size(); ++i) {
                if (dnode.count(toUpper(an[i]))) throw std::runtime_error(a.name + ": node " + an[i] + " is digital; this port is analog");
                x.analog.push_back(analogNode(an[i], a.name));
                (adc ? x.out : x.in).push_back(digital(dn[i], a.name, false));
                if (!adc) {  // the output: an ideal source to ground, ramped by the simulation (ngspice: a#branch)
                    VoltageSource v{};
                    v.name = toUpper(a.name) + "#" + std::to_string(i);
                    v.nodePositive = x.analog.back();
                    v.nodeNegative = 0;
                    x.vsrc.push_back((int)result_.netlist.globalBlock.voltageSources.size());
                    result_.netlist.globalBlock.voltageSources.push_back(v);
                }
            }
            allowed = adc ? std::set<std::string>{"IN_LOW", "IN_HIGH", "RISE_DELAY", "FALL_DELAY", "IN_LOAD", "FAMILY"}
                          : std::set<std::string>{"OUT_LOW", "OUT_HIGH", "OUT_UNDEF", "T_RISE", "T_FALL", "INPUT_LOAD", "FAMILY"};
            break;
        }
        }
        for (const auto& [key, v] : c->params)
            if (!allowed.count(key)) throw std::runtime_error("model " + c->name + ": parameter " + key + " is not a " + c->type + " parameter");
        x.inertial = c->get("INERTIAL_DELAY", 0.0) != 0.0;
        if (x.inertial && x.kind == Kind::TRI)  // d_tristate's runs its state and strength as two inertial channels
            throw std::runtime_error("model " + c->name + ": inertial_delay on a d_tristate is not supported yet");
        const double d = c->get("DELAY", 1e-9);  // d_tristate
        x.rise = x.kind == Kind::TRI ? d : c->get(x.kind == Kind::OPENC ? "OPEN_DELAY" : "RISE_DELAY", 1e-9);
        x.fall = x.kind == Kind::TRI ? d : c->get(x.kind == Kind::OPENE ? "OPEN_DELAY" : "FALL_DELAY", 1e-9);
        x.clk = c->get(x.kind == Kind::DLATCH || x.kind == Kind::SRLATCH ? "ENABLE_DELAY" : "CLK_DELAY", 1e-9);
        x.data = x.kind == Kind::RAM ? c->get("READ_DELAY", 100e-9) : c->get(x.kind == Kind::DLATCH ? "DATA_DELAY" : "SR_DELAY", 1e-9);
        x.set = c->get("SET_DELAY", 1e-9); x.reset = c->get("RESET_DELAY", 1e-9);
        const double ic = c->get("IC", x.kind == Kind::RAM ? 2.0 : 0.0);  // d_ram's ic defaults to U
        x.ic = ic == 1.0 ? xspice::L1 : ic == 0.0 ? xspice::L0 : xspice::LU;
        x.lo = c->get("IN_LOW", 0.1); x.hi = c->get("IN_HIGH", 0.9);
        x.vlo = c->get("OUT_LOW", 0.0); x.vhi = c->get("OUT_HIGH", 1.0);
        // dac_bridge: with out_low and out_high given and out_undef not, U is their mean
        x.vun = c->params.count("OUT_LOW") && c->params.count("OUT_HIGH") && !c->params.count("OUT_UNDEF") ? (x.vlo + x.vhi) / 2.0
                                                                                                         : c->get("OUT_UNDEF", 0.5);
        x.trise = c->get("T_RISE", 1e-9); x.tfall = c->get("T_FALL", 1e-9);
        if (!(x.hi > x.lo)) throw std::runtime_error("model " + c->name + ": in_high must be > in_low");
        if (x.kind == Kind::DAC && !(x.trise > 0.0 && x.tfall > 0.0 && x.vhi != x.vlo))
            throw std::runtime_error("model " + c->name + ": needs t_rise, t_fall > 0 and out_high != out_low");
        net->insts.push_back(std::move(x));
    }
    // A digital net an X line numbered as a node: unnamed (no analog output), held at 0 V (nothing else is on it)
    for (int idx : orphan) {
        if (analogTop_.count(idx) || analogGlobal_.count(idx)) continue;  // an adc/dac port made it analog after all
        for (auto* names : {&nodeMap_, &result_.nodeIndex})
            for (auto it = names->begin(); it != names->end();) it = it->second == idx ? names->erase(it) : std::next(it);
        Resistor r{};
        r.name = "#XSPICE_NET" + std::to_string(idx); r.nodeTerminal1 = idx; r.nodeTerminal2 = 0; r.resistance_ohms = 1.0;
        result_.netlist.globalBlock.resistors.push_back(r);
    }
    if (!net->insts.empty()) result_.netlist.digital = net;
}

// An XSPICE analog code model (physics/xspice_analog.h): its ports walked by the model's port table (a
// differential type takes two nodes, a vector port [ ... ]), each output a behavioural source over the inputs.
// A current input is a zero-volt source across its port (ngspice MIF), read through its branch current.
void SPICEParser::resolveXspiceAnalog(const PendingA& a, const ModelCard& c,
                                      const std::function<int(const std::string&, const std::string&)>& analogNode) {
    using xspice::AKind;
    // dir: i(nput), o(utput), b(oth: a conductance port, g or gd), h (an hd port: a voltage output reading its own
    // current); allow: a b port's types if more than def; nullOk: NULL allowed
    struct Spec { char dir; const char* def; bool vec; const char* allow = nullptr; bool nullOk = false; };
    static const std::map<std::string, std::pair<AKind, std::vector<Spec>>> table = {
        {"GAIN", {AKind::GAIN, {{'i', "v", false}, {'o', "v", false}}}},
        {"SUMMER", {AKind::SUMMER, {{'i', "v", true}, {'o', "v", false}}}},
        {"MULT", {AKind::MULT, {{'i', "v", true}, {'o', "v", false}}}},
        {"DIVIDE", {AKind::DIVIDE, {{'i', "v", false}, {'i', "v", false}, {'o', "v", false}}}},
        {"LIMIT", {AKind::LIMIT, {{'i', "v", false}, {'o', "v", false}}}},
        {"CLIMIT", {AKind::CLIMIT, {{'i', "v", false}, {'i', "v", false}, {'i', "v", false}, {'o', "v", false}}}},
        {"PWL", {AKind::PWL, {{'i', "v", false}, {'o', "v", false}}}},
        {"ASWITCH", {AKind::ASWITCH, {{'i', "v", false}, {'b', "gd", false}}}},
        {"INT", {AKind::INT, {{'i', "v", false}, {'o', "v", false}}}},
        {"D_DT", {AKind::D_DT, {{'i', "v", false}, {'o', "v", false}}}},
        {"S_XFER", {AKind::S_XFER, {{'i', "v", false}, {'o', "v", false}}}},
        {"SLEW", {AKind::SLEW, {{'i', "v", false}, {'o', "v", false}}}},
        {"HYST", {AKind::HYST, {{'i', "v", false}, {'o', "v", false}}}},
        {"SINE", {AKind::SINE, {{'i', "v", false}, {'o', "v", false}}}},
        {"TRIANGLE", {AKind::TRIANGLE, {{'i', "v", false}, {'o', "v", false}}}},
        {"SQUARE", {AKind::SQUARE, {{'i', "v", false}, {'o', "v", false}}}},
        {"ZENER", {AKind::ZENER, {{'b', "gd", false}}}},
        {"MULTI_INPUT_PWL", {AKind::MIPWL, {{'i', "vd", true}, {'o', "vd", false}}}},
        {"OTA", {AKind::OTA, {{'b', "gd", false}, {'b', "g", false}}}},
        {"CORE", {AKind::CORE, {{'b', "gd", false, "g gd"}}}},
        {"LCOUPLE", {AKind::LCOUPLE, {{'h', "hd", false, "h hd"}, {'h', "hd", false}}}},
        {"XFER", {AKind::XFER, {{'i', "v", false}, {'o', "v", false}}}},
        {"PWLTS", {AKind::PWLTS, {{'o', "v", false}}}},
        {"FILESOURCE", {AKind::FILESRC, {{'o', "v", true}}}},
        {"MEMRISTOR", {AKind::MEMRISTOR, {{'b', "gd", false}}}},
        {"PSWITCH", {AKind::PSWITCH, {{'b', "gd", false, "g gd"}, {'b', "gd", false}}}},
        {"SIDIODE", {AKind::SIDIODE, {{'b', "gd", false}}}},
        {"CAPACITORIC", {AKind::CAPIC, {{'h', "hd", false}}}},
        {"INDUCTORIC", {AKind::INDIC, {{'b', "gd", false}}}},
        {"POTENTIOMETER", {AKind::POT, {{'b', "g", false}, {'b', "g", false}, {'b', "g", false}}}},
        {"ONESHOT", {AKind::ONESHOT, {{'i', "v", false}, {'i', "v", false, nullptr, true}, {'i', "v", false, nullptr, true}, {'o', "v", false}}}},
        {"ILIMIT", {AKind::ILIMIT, {{'i', "v", false}, {'b', "g", false, "g gd", true}, {'b', "g", false, "g gd", true}, {'b', "g", false, "g gd"}}}}};
    const auto& [kind, specs] = table.at(c.type);
    struct Port { std::string type; int n1 = 0, n2 = 0; std::string vname; };
    std::vector<std::vector<Port>> ports;  // per spec, its elements
    size_t p = 0;
    const auto& w = a.raw;
    auto one = [&](const Spec& sp) {
        Port pt;
        pt.type = sp.def;
        if (p < w.size() && w[p][0] == '%') {
            pt.type = w[p].substr(1);
            for (auto& ch : pt.type) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            ++p;
        }
        static const std::set<std::string> in = {"v", "vd", "i", "id", "vnam"}, out = {"v", "vd", "i", "id"};
        if (sp.nullOk && p < w.size() && toUpper(w[p]) == "NULL") { ++p; pt.type = "null"; return pt; }
        if (!(sp.dir == 'h' || sp.dir == 'b' ? (sp.allow ? (" " + std::string(sp.allow) + " ").find(" " + pt.type + " ") != std::string::npos : pt.type == sp.def)
                            : (sp.dir == 'i' ? in : out).count(pt.type)))
            throw std::runtime_error(a.name + ": port type %" + pt.type + " is not allowed (or not supported yet) on this " + c.type + " port");
        const bool diff = pt.type == "vd" || pt.type == "id" || pt.type == "gd" || pt.type == "hd";
        auto node = [&]() {
            if (p >= w.size() || w[p] == "[" || w[p] == "]" || w[p][0] == '%') throw std::runtime_error(a.name + ": missing a node for " + c.type);
            const std::string n = w[p++];
            if (toUpper(n) == "NULL" || n[0] == '~') throw std::runtime_error(a.name + ": " + n + " on an analog port");
            return n;
        };
        if (pt.type == "vnam") { pt.vname = toUpper(node()); return pt; }
        pt.n1 = analogNode(node(), a.name);
        if (diff) pt.n2 = analogNode(node(), a.name);
        return pt;
    };
    for (const auto& sp : specs) {
        std::vector<Port> el;
        if (sp.vec) {
            if (p >= w.size() || w[p] != "[") throw std::runtime_error(a.name + ": " + c.type + " takes a [ ... ] vector of inputs");
            for (++p; p < w.size() && w[p] != "]";) el.push_back(one(sp));
            if (p >= w.size()) throw std::runtime_error(a.name + ": '[' without ']'");
            ++p;
            if (el.empty()) throw std::runtime_error(a.name + ": no inputs");
        } else el.push_back(one(sp));
        ports.push_back(el);
    }
    if (p != w.size()) throw std::runtime_error(a.name + ": " + c.type + " takes " + std::to_string(specs.size()) + " ports; '" + w[p] + "' is one too many");

    // parameters: the card's, ngspice's defaults (ifspec.ifs)
    auto m = std::make_shared<xspice::AnalogModel>();
    m->kind = kind;
    std::set<std::string> allowed;
    auto num = [&](const char* k, double def) { allowed.insert(k); return c.get(k, def); };
    auto flag = [&](const char* k, bool def) { allowed.insert(k); return c.params.count(k) ? c.params.at(k) != 0.0 : def; };
    auto vec = [&](const char* k, size_t n, double def) {
        allowed.insert(k);
        const auto it = c.vectors.find(k);
        std::vector<double> v = it == c.vectors.end() ? std::vector<double>() : it->second;
        if (v.empty() && c.params.count(k)) v = {c.params.at(k)};
        if (v.empty()) v.assign(n, def);
        if (v.size() != n) throw std::runtime_error("model " + c.name + ": " + k + " has " + std::to_string(v.size()) + " values for " + std::to_string(n) + " inputs");
        return v;
    };
    const size_t nin = ports[0].size();
    switch (kind) {
    case AKind::GAIN:
        m->inOffset = num("IN_OFFSET", 0.0); m->gain = num("GAIN", 1.0); m->outOffset = num("OUT_OFFSET", 0.0);
        for (const char* k : {"NOISE_VOLTAGE", "NOISE_CURRENT", "NOISE_CORNER", "NOISE_EXPONENT"}) allowed.insert(k);
        if (c.get("NOISE_VOLTAGE", 0.0) != 0.0 || c.get("NOISE_CURRENT", 0.0) != 0.0)
            throw std::runtime_error("model " + c.name + ": the gain code model's noise is not supported yet");
        break;
    case AKind::SUMMER: case AKind::MULT:
        m->inOffsets = vec("IN_OFFSET", nin, 0.0); m->inGains = vec("IN_GAIN", nin, 1.0);
        m->outGain = num("OUT_GAIN", 1.0); m->outOffset = num("OUT_OFFSET", 0.0);
        break;
    case AKind::DIVIDE:
        m->numOffset = num("NUM_OFFSET", 0.0); m->numGain = num("NUM_GAIN", 1.0);
        m->denOffset = num("DEN_OFFSET", 0.0); m->denGain = num("DEN_GAIN", 1.0);
        m->denLower = num("DEN_LOWER_LIMIT", 1e-10); m->denDomain = num("DEN_DOMAIN", 1e-16); m->fraction = flag("FRACTION", false);
        m->outGain = num("OUT_GAIN", 1.0); m->outOffset = num("OUT_OFFSET", 0.0);
        break;
    case AKind::LIMIT: case AKind::CLIMIT:
        m->inOffset = num("IN_OFFSET", 0.0); m->gain = num("GAIN", 1.0);
        m->range = num("LIMIT_RANGE", 1e-6); m->fraction = flag("FRACTION", false);
        if (kind == AKind::LIMIT) {
            allowed.insert("OUT_LOWER_LIMIT"); allowed.insert("OUT_UPPER_LIMIT");
            if (!c.params.count("OUT_LOWER_LIMIT") || !c.params.count("OUT_UPPER_LIMIT"))
                throw std::runtime_error("model " + c.name + ": limit needs out_lower_limit and out_upper_limit");
            m->lo = c.params.at("OUT_LOWER_LIMIT"); m->hi = c.params.at("OUT_UPPER_LIMIT");
            if (m->fraction) { m->range *= m->hi - m->lo; m->fraction = false; }
        } else { m->upperDelta = num("UPPER_DELTA", 0.0); m->lowerDelta = num("LOWER_DELTA", 0.0); }
        break;
    case AKind::PWL: case AKind::PWLTS: {
        allowed.insert("X_ARRAY"); allowed.insert("Y_ARRAY");
        const auto xi = c.vectors.find("X_ARRAY"), yi = c.vectors.find("Y_ARRAY");
        if (xi == c.vectors.end() || yi == c.vectors.end() || xi->second.size() != yi->second.size() || xi->second.size() < 2)
            throw std::runtime_error("model " + c.name + ": pwl needs x_array and y_array of the same size, 2 points at least");
        const auto& x = xi->second; const auto& y = yi->second;
        for (size_t i = 1; i < x.size(); ++i)
            if (!(x[i] > x[i - 1])) throw std::runtime_error("model " + c.name + ": pwl x_array must increase");
        const size_t n = x.size() + 2;
        m->x.assign(n, 0.0); m->y.assign(n, 0.0);
        for (size_t i = 1; i + 1 < n; ++i) { m->x[i] = x[i - 1]; m->y[i] = y[i - 1]; }
        m->x[0] = 2 * m->x[1] - m->x[2]; m->x[n - 1] = 2 * m->x[n - 2] - m->x[n - 3];
        if (flag("LIMIT", false)) { m->y[0] = m->y[1]; m->y[n - 1] = m->y[n - 2]; }
        else { m->y[0] = 2 * m->y[1] - m->y[2]; m->y[n - 1] = 2 * m->y[n - 2] - m->y[n - 3]; }
        m->domain = num("INPUT_DOMAIN", 0.01); m->fraction = flag("FRACTION", true);
        if (kind == AKind::PWLTS)  // ngspice cm_pwlts: breakpoints at each corner and either side of its smoothing (after t = 0)
            for (size_t i = 1; i + 1 < n; ++i) {
                const double dt = m->fraction ? std::min(m->x[i] - m->x[i - 1], m->x[i + 1] - m->x[i]) * m->domain : m->domain;
                if (m->x[i] - dt <= 0.0) continue;
                for (double b : {m->x[i] - dt, m->x[i], m->x[i] + dt}) m->bkpts.push_back(b);
            }
        std::sort(m->bkpts.begin(), m->bkpts.end());
        break;
    }
    case AKind::INT: case AKind::D_DT:
        m->gain = num("GAIN", 1.0);
        if (kind == AKind::INT) { m->inOffset = num("IN_OFFSET", 0.0); m->outIc = num("OUT_IC", 0.0); }
        else m->outOffset = num("OUT_OFFSET", 0.0);
        m->range = num("LIMIT_RANGE", 1e-6);
        allowed.insert("OUT_LOWER_LIMIT"); allowed.insert("OUT_UPPER_LIMIT");
        if (!c.params.count("OUT_LOWER_LIMIT") || !c.params.count("OUT_UPPER_LIMIT"))  // ngspice: no defaults
            throw std::runtime_error("model " + c.name + ": " + c.type + " needs out_lower_limit and out_upper_limit");
        m->lo = c.params.at("OUT_LOWER_LIMIT"); m->hi = c.params.at("OUT_UPPER_LIMIT");
        break;
    case AKind::S_XFER: {
        m->inOffset = num("IN_OFFSET", 0.0);
        allowed.insert("NUM_COEFF"); allowed.insert("DEN_COEFF"); allowed.insert("INT_IC");
        auto coeffs = [&](const char* k) {
            const auto it = c.vectors.find(k);
            if (it != c.vectors.end()) return it->second;
            if (c.params.count(k)) return std::vector<double>{c.params.at(k)};
            throw std::runtime_error("model " + c.name + ": s_xfer needs " + std::string(k));
        };
        const auto nc = coeffs("NUM_COEFF"), dc = coeffs("DEN_COEFF");  // highest power first
        if (nc.size() > dc.size() || dc.size() < 2)
            throw std::runtime_error("model " + c.name + ": s_xfer needs a denominator of order 1 at least, not below the numerator's");
        const double f = num("DENORMALIZED_FREQ", 1.0);
        for (size_t i = 0; i < nc.size(); ++i) m->num.push_back(nc[nc.size() - 1 - i] / std::pow(f, (double)i));
        for (size_t i = 0; i < dc.size(); ++i) m->den.push_back(dc[dc.size() - 1 - i] / std::pow(f, (double)i));
        const double top = m->den.back();
        if (top == 0.0) throw std::runtime_error("model " + c.name + ": s_xfer's highest denominator coefficient is 0");
        for (auto& d : m->den) d /= top;
        m->gain = num("GAIN", 1.0) / top;
        m->intIc.assign(dc.size(), 0.0);
        if (const auto it = c.vectors.find("INT_IC"); it != c.vectors.end() || c.params.count("INT_IC")) {
            const std::vector<double> ic = it != c.vectors.end() ? it->second : std::vector<double>{c.params.at("INT_IC")};
            if (ic.size() != dc.size() - 1)
                throw std::runtime_error("model " + c.name + ": s_xfer int_ic needs " + std::to_string(dc.size() - 1) + " values");
            for (size_t i = 0; i + 1 < dc.size(); ++i) m->intIc[i] = ic[dc.size() - 2 - i];
        }
        break;
    }
    case AKind::SLEW:
        m->riseSlope = num("RISE_SLOPE", 1e9); m->fallSlope = num("FALL_SLOPE", 1e9);
        if (!(m->riseSlope > 0.0 && m->fallSlope > 0.0)) throw std::runtime_error("model " + c.name + ": slew slopes must be > 0");
        break;
    case AKind::SINE: case AKind::TRIANGLE: case AKind::SQUARE: {
        auto arr = [&](const char* k, std::vector<double> def) {
            allowed.insert(k);
            const auto it = c.vectors.find(k);
            return it != c.vectors.end() ? it->second : def;
        };
        m->cntl = arr("CNTL_ARRAY", {0.0, 1.0}); m->freq = arr("FREQ_ARRAY", {1e3, 2e3});
        if (m->cntl.size() != m->freq.size() || m->cntl.size() < 2)
            throw std::runtime_error("model " + c.name + ": cntl_array and freq_array need the same size, 2 at least");
        m->outLow = num("OUT_LOW", -1.0); m->outHigh = num("OUT_HIGH", 1.0);
        if (kind != AKind::SINE) m->duty = num("DUTY_CYCLE", 0.5);
        if (kind == AKind::SQUARE) { m->tRise = num("RISE_TIME", 1e-9); m->tFall = num("FALL_TIME", 1e-9); }
        if (!(m->duty > 0.0 && m->duty < 1.0)) throw std::runtime_error("model " + c.name + ": duty_cycle must be in (0, 1)");
        break;
    }
    case AKind::ZENER:
        allowed.insert("V_BREAKDOWN");
        if (!c.params.count("V_BREAKDOWN")) throw std::runtime_error("model " + c.name + ": zener needs v_breakdown");
        m->vBreak = c.params.at("V_BREAKDOWN");
        m->iBreak = num("I_BREAKDOWN", 2e-2); m->rBreak = num("R_BREAKDOWN", 1.0); m->iRev = num("I_REV", 1e-6);
        m->iSat = num("I_SAT", 1e-12); m->nFwd = num("N_FORWARD", 1.0);
        num("LIMIT_SWITCH", 0.0);  // ngspice's input limiting: the iteration path only
        if (!(m->vBreak > 0.0 && m->iBreak > 0.0 && m->rBreak > 0.0 && m->iSat > 0.0 && m->nFwd > 0.0))
            throw std::runtime_error("model " + c.name + ": zener needs v_breakdown, i_breakdown, r_breakdown, i_sat, n_forward > 0");
        break;
    case AKind::HYST:
        m->inLow = num("IN_LOW", 0.0); m->inHigh = num("IN_HIGH", 1.0); m->hyst = num("HYST", 0.1);
        m->lo = num("OUT_LOWER_LIMIT", 0.0); m->hi = num("OUT_UPPER_LIMIT", 1.0);
        m->domain = num("INPUT_DOMAIN", 0.01); m->fraction = flag("FRACTION", true);
        if (!(m->inHigh > m->inLow)) throw std::runtime_error("model " + c.name + ": hyst needs in_high > in_low");
        break;
    case AKind::MIPWL: {
        allowed.insert("X"); allowed.insert("Y");
        const auto xi = c.vectors.find("X"), yi = c.vectors.find("Y");
        if (xi == c.vectors.end() || yi == c.vectors.end() || xi->second.size() != yi->second.size() || xi->second.size() < 2)
            throw std::runtime_error("model " + c.name + ": multi_input_pwl needs x and y of the same size, 2 at least");
        m->x = xi->second; m->y = yi->second;
        for (size_t i = 1; i < m->x.size(); ++i)
            if (!(m->x[i] > m->x[i - 1])) throw std::runtime_error("model " + c.name + ": multi_input_pwl x must increase");
        const std::string g = c.strings.count("MODEL") ? c.strings.at("MODEL") : "and";
        static const std::vector<std::string> gates = {"and", "nand", "or", "nor"};
        const auto gi = std::find(gates.begin(), gates.end(), g);
        if (gi == gates.end()) throw std::runtime_error("model " + c.name + ": multi_input_pwl model " + g + " is not and, nand, or, nor");
        m->gate = static_cast<int>(gi - gates.begin());
        break;
    }
    case AKind::CORE: {
        const double mode = num("MODE", 1.0);
        m->coreHyst = mode == 2.0;
        m->domain = num("INPUT_DOMAIN", 0.01); m->fraction = flag("FRACTION", true);
        m->inLow = num("IN_LOW", 0.0); m->inHigh = num("IN_HIGH", 1.0); m->hyst = num("HYST", 0.1);
        m->lo = num("OUT_LOWER_LIMIT", 0.0); m->hi = num("OUT_UPPER_LIMIT", 1.0);
        allowed.insert("H_ARRAY"); allowed.insert("B_ARRAY"); allowed.insert("AREA"); allowed.insert("LENGTH");
        const auto hi = c.vectors.find("H_ARRAY"), bi = c.vectors.find("B_ARRAY");
        if (hi == c.vectors.end() || bi == c.vectors.end() || hi->second.size() != bi->second.size() || hi->second.size() < 2)
            throw std::runtime_error("model " + c.name + ": core needs h_array and b_array of the same size, 2 points at least");
        if (!c.params.count("AREA") || !c.params.count("LENGTH") || !(c.params.at("AREA") > 0.0) || !(c.params.at("LENGTH") > 0.0))
            throw std::runtime_error("model " + c.name + ": core needs area > 0 and length > 0");
        m->x = hi->second; m->y = bi->second; m->area = c.params.at("AREA"); m->length = c.params.at("LENGTH");
        if (m->coreHyst) {  // ngspice requires the pwl parameters in hysteresis mode too, and ignores them
            if (!(m->inHigh > m->inLow)) throw std::runtime_error("model " + c.name + ": core (hysteresis) needs in_high > in_low");
            break;
        }
        for (size_t i = 1; i < m->x.size(); ++i) {
            if (!(m->x[i] > m->x[i - 1])) throw std::runtime_error("model " + c.name + ": core h_array must increase");
            if (!m->fraction && m->x[i] - m->x[i - 1] < 2.0 * m->domain)  // ngspice then gives no output at all
                throw std::runtime_error("model " + c.name + ": core h_array steps must be at least 2 input_domain (fraction false)");
        }
        break;
    }
    case AKind::LCOUPLE:
        m->gain = num("NUM_TURNS", 1.0);
        break;
    case AKind::XFER: {
        // ngspice cm_xfer's table: span numbers per entry, the frequency [Hz] then a pair at offset (magnitude in dB
        // and phase in degrees by default; r_i: real / imaginary; db false: magnitude; rad: phase in radians)
        allowed.insert("TABLE");
        if (c.strings.count("FILE") || c.params.count("FILE")) throw std::runtime_error("model " + c.name + ": xfer's Touchstone file is not supported yet");
        const double spanD = num("SPAN", 3.0), offD = num("OFFSET", 1.0);
        const bool ri = flag("R_I", false), db = flag("DB", true), rad = flag("RAD", false);
        if (spanD != std::floor(spanD) || offD != std::floor(offD) || offD < 1.0 || spanD < offD + 2.0)
            throw std::runtime_error("model " + c.name + ": xfer needs integer offset >= 1 and span >= offset + 2");
        const size_t span = static_cast<size_t>(spanD), off = static_cast<size_t>(offD);
        const auto ti = c.vectors.find("TABLE");
        if (ti == c.vectors.end() || ti->second.empty() || ti->second.size() % span)
            throw std::runtime_error("model " + c.name + ": xfer needs a table of whole entries (" + std::to_string(span) + " numbers each)");
        const auto& tb = ti->second;
        for (size_t i = 0; i < tb.size(); i += span) {
            if (tb[i] < 0.0 || (i && tb[i] < tb[i - span])) throw std::runtime_error("model " + c.name + ": xfer frequencies must be >= 0 and not decrease");
            double a = tb[i + off], b = tb[i + off + 1];
            m->x.push_back(tb[i] * 2.0 * M_PI);
            if (ri) { m->sx.push_back({a, b}); continue; }
            if (db) a = std::pow(10.0, a / 20.0);
            if (!rad) b *= 2.0 * M_PI / 360.0;
            m->sx.push_back(std::polar(a, b));
        }
        break;
    }
    case AKind::FILESRC: {
        // ngspice cm_filesource: rows "time value...", lines from * # ; skipped, values split at blanks and commas;
        // times scaled, then offset (or each relative to the last); values scaled, then offset, per column.
        // ngspice skips a bad time and keeps a short row; both are errors here.
        allowed.insert("AMPLOFFSET"); allowed.insert("AMPLSCALE");
        auto arr = [&](const char* k) {
            if (c.vectors.count(k)) return c.vectors.at(k);
            if (c.params.count(k)) return std::vector<double>{c.params.at(k)};
            throw std::runtime_error("model " + c.name + ": filesource needs " + std::string(k == std::string("AMPLOFFSET") ? "amploffset" : "amplscale"));
        };
        const auto off = arr("AMPLOFFSET"), scale = arr("AMPLSCALE");
        const double toff = num("TIMEOFFSET", 0.0), tscale = num("TIMESCALE", 1.0);
        const bool rel = flag("TIMERELATIVE", false);
        m->fraction = flag("AMPLSTEP", false);  // hold each row until the next
        const size_t width = ports[0].size();
        m->width = static_cast<int>(width);
        const std::string file = c.strings.count("FILE") ? c.strings.at("FILE") : "filesource.txt";
        std::ifstream in(dataPath(file));
        if (!in) throw std::runtime_error("model " + c.name + ": cannot open filesource file " + file);
        std::string line;
        double tprev = toff;
        for (int n = 1; std::getline(in, line); ++n) {
            const size_t b = line.find_first_not_of(" \t\r");
            if (b == std::string::npos || line[b] == '*' || line[b] == '#' || line[b] == ';') continue;
            for (char& ch : line) if (ch == ',') ch = ' ';
            std::istringstream ss(line);
            std::vector<std::string> tok;
            for (std::string w; ss >> w;) tok.push_back(w);
            const std::string where = "model " + c.name + ": " + file + " line " + std::to_string(n) + ": ";
            if (tok.size() < width + 1) throw std::runtime_error(where + "expected a time and " + std::to_string(width) + " values");
            double t = parseValue(tok[0]) * tscale;
            t = rel ? t + tprev : t + toff;
            tprev = t;
            if (!m->x.empty() && !(t > m->x.back())) throw std::runtime_error(where + "times must increase");
            m->x.push_back(t);
            for (size_t j = 0; j < width; ++j) {
                double d = parseValue(tok[j + 1]);
                if (j < scale.size()) d *= scale[j];
                if (j < off.size()) d += off[j];
                m->y.push_back(d);
            }
        }
        if (m->x.size() < 2) throw std::runtime_error("model " + c.name + ": filesource file " + file + " needs 2 rows at least");
        break;
    }
    case AKind::MEMRISTOR:
        m->rMin = num("RMIN", 10.0); m->rMax = num("RMAX", 1e4); m->outIc = num("RINIT", 7000.0);
        m->mAlpha = num("ALPHA", 0.0); m->mBeta = num("BETA", 1.0); m->mVt = num("VT", 0.0);
        if (!(m->rMin > 0.0 && m->rMax > m->rMin && m->outIc > 0.0)) throw std::runtime_error("model " + c.name + ": memristor needs 0 < rmin < rmax and rinit > 0");
        break;
    case AKind::PSWITCH:
        m->cntlOff = num("CNTL_OFF", 0.0); m->cntlOn = num("CNTL_ON", 1.0); m->logR = flag("LOG", true);
        m->rOff = num("R_OFF", 1e12); m->rOn = num("R_ON", 1.0); m->rCntl = num("R_CNTL_IN", 1e12);
        if (!(m->rOn > 0.0 && m->rOff > 0.0 && m->rCntl > 0.0)) throw std::runtime_error("model " + c.name + ": pswitch needs r_on, r_off, r_cntl_in > 0");
        break;
    case AKind::SIDIODE:
        m->sRon = num("RON", 1.0); m->sRoff = num("ROFF", 1.0); m->sRrev = num("RREV", 0.0);
        m->vFwd = num("VFWD", 0.0); m->vRev = num("VREV", 1e30); m->iLim = num("ILIMIT", 1e30); m->revILim = num("REVILIMIT", 1e30);
        m->eps = num("EPSILON", 0.0); m->revEps = num("REVEPSILON", 0.0);
        if (!(m->sRon > 0.0 && m->sRoff > 0.0 && m->sRrev >= 0.0 && m->iLim > 0.0 && m->revILim > 0.0 && m->eps >= 0.0 && m->revEps >= 0.0))
            throw std::runtime_error("model " + c.name + ": sidiode needs ron, roff, ilimit, revilimit > 0 and rrev, epsilon, revepsilon >= 0");
        break;
    case AKind::CAPIC: case AKind::INDIC: {
        const char* k = kind == AKind::CAPIC ? "C" : "L";
        allowed.insert(k);
        if (!c.params.count(k)) throw std::runtime_error("model " + c.name + ": " + c.type + " needs " + (kind == AKind::CAPIC ? "c" : "l"));
        m->gain = c.params.at(k);
        if (!(m->gain > 0.0)) throw std::runtime_error("model " + c.name + ": " + c.type + " needs " + (kind == AKind::CAPIC ? "c" : "l") + " > 0");
        m->outIc = num("IC", 0.0);
        break;
    }
    case AKind::POT: {
        // ngspice: position kept inside (0, 1) (its [0, 1] limit is not checked); log: r0-wiper is r / 10^(position log_multiplier)
        double pos = num("POSITION", 0.5);
        pos = pos <= 0.0 ? 1e-9 : pos >= 1.0 ? 0.999999999 : pos;
        const double r = num("R", 1e5), mult = num("LOG_MULTIPLIER", 1.0);
        m->rOn = flag("LOG", false) ? r / std::pow(10.0, pos * mult) : pos * r;
        m->rOff = r - m->rOn;
        if (!(m->rOn > 0.0 && m->rOff > 0.0)) throw std::runtime_error("model " + c.name + ": potentiometer needs r > 0 and both sections > 0");
        break;
    }
    case AKind::ONESHOT: {
        auto arr = [&](const char* k, std::vector<double> def) {
            allowed.insert(k);
            const auto it = c.vectors.find(k);
            return it != c.vectors.end() ? it->second : def;
        };
        m->x = arr("CNTL_ARRAY", {0.0, 1.0}); m->y = arr("PW_ARRAY", {1e-6, 0.9999999});
        if (m->x.size() != m->y.size() || m->x.size() < 2)
            throw std::runtime_error("model " + c.name + ": cntl_array and pw_array need the same size, 2 at least");
        for (size_t i = 0; i < m->y.size(); ++i) {
            if (i && !(m->x[i] > m->x[i - 1])) throw std::runtime_error("model " + c.name + ": cntl_array must increase");
            if (!(m->y[i] >= 0.0)) throw std::runtime_error("model " + c.name + ": pw_array values must be >= 0");
        }
        m->trig = num("CLK_TRIG", 0.5); m->posEdge = flag("POS_EDGE_TRIG", true);
        m->outLow = num("OUT_LOW", 0.0); m->outHigh = num("OUT_HIGH", 1.0);
        m->tRise = num("RISE_TIME", 1e-9); m->tFall = num("FALL_TIME", 1e-9);
        m->rDelay = num("RISE_DELAY", 1e-9); m->fDelay = num("FALL_DELAY", 1e-9); m->retrig = flag("RETRIG", false);
        m->cntlNull = ports[1][0].type == "null"; m->clearNull = ports[2][0].type == "null";
        m->intIc = {-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, m->outLow, 0.0};
        break;
    }
    case AKind::ILIMIT: {
        m->inOffset = num("IN_OFFSET", 0.0); m->gain = num("GAIN", 1.0);
        m->rSource = num("R_OUT_SOURCE", 1.0); m->rSink = num("R_OUT_SINK", 1.0);
        m->iSource = num("I_LIMIT_SOURCE", 10e-3); m->iSink = num("I_LIMIT_SINK", 10e-3);
        m->range = num("V_PWR_RANGE", 1e-6); m->iSourceRange = num("I_SOURCE_RANGE", 1e-9);
        m->iSinkRange = num("I_SINK_RANGE", 1e-9); m->rDomain = num("R_OUT_DOMAIN", 1e-9);
        if (!(m->rSource >= 1e-9 && m->rSource <= 1e9 && m->rSink >= 1e-9 && m->rSink <= 1e9))
            throw std::runtime_error("model " + c.name + ": ilimit r_out_source and r_out_sink must be in [1e-9, 1e9]");
        if (!(m->iSource >= 1e-12 && m->iSink >= 1e-12 && m->range >= 1e-15 && m->iSourceRange >= 1e-15 && m->iSinkRange >= 1e-15 && m->rDomain >= 1e-15))
            throw std::runtime_error("model " + c.name + ": ilimit limits must be >= 1e-12 and ranges >= 1e-15");
        m->posNull = ports[1][0].type == "null"; m->negNull = ports[2][0].type == "null";
        break;
    }
    case AKind::OTA: {
        m->gain = num("GM", 1e-3); m->rOff = num("ROUT", 1e12); m->rOn = num("RIN", 1e12);
        double noise = 0.0;
        for (const char* k : {"EN", "IN_NOISE", "ENK", "INK", "INCM", "INCMK"}) noise += std::abs(num(k, 0.0));
        flag("NOISE_PROGRAMMATIC", true);
        if (noise != 0.0) throw std::runtime_error("model " + c.name + ": the ota code model's noise is not supported yet");
        if (!(m->gain >= 1e-15 && m->rOff > 0.0 && m->rOn > 0.0)) throw std::runtime_error("model " + c.name + ": ota needs gm >= 1e-15, rout > 0, rin > 0");
        break;
    }
    case AKind::ASWITCH:
        m->cntlOff = num("CNTL_OFF", 0.0); m->cntlOn = num("CNTL_ON", 1.0);
        m->rOff = num("R_OFF", 1e12); m->rOn = num("R_ON", 1.0);
        m->logR = flag("LOG", true); m->limitR = flag("LIMIT", false);
        if (std::abs(m->cntlOn - m->cntlOff) < 1e-12) throw std::runtime_error("model " + c.name + ": aswitch needs |cntl_on - cntl_off| >= 1e-12");
        if (m->logR && !(m->rOff > 0.0)) throw std::runtime_error("model " + c.name + ": aswitch log needs r_off > 0");
        break;
    }
    for (const auto& [k, v] : c.params)
        if (!allowed.count(k)) throw std::runtime_error("model " + c.name + ": parameter " + k + " is not a " + c.type + " parameter");
    for (const auto& [k, v] : c.vectors)
        if (!allowed.count(k)) throw std::runtime_error("model " + c.name + ": parameter " + k + " is not a " + c.type + " parameter");

    // inputs -> references (one per input element, in port order)
    auto& blk = result_.netlist.globalBlock;
    std::vector<Behavioural::Ref> refs;
    int sense = 0;
    for (size_t s = 0; s < specs.size(); ++s) {
        if (specs[s].dir != 'i') continue;
        for (const auto& pt : ports[s]) {
            Behavioural::Ref r;
            if (pt.type == "v" || pt.type == "vd" || pt.type == "null") { r.kind = 0; r.a = pt.n1; r.b = pt.n2; }  // NULL: 0 - 0
            else {
                r.kind = 1;
                if (pt.type == "vnam") {
                    int idx = -1;
                    for (size_t k = 0; k < blk.voltageSources.size(); ++k) if (toUpper(blk.voltageSources[k].name) == pt.vname) idx = (int)k;
                    if (idx < 0) throw std::runtime_error(a.name + ": %vnam " + pt.vname + " is no top-level voltage source");
                    r.a = idx;
                } else {  // %i / %id: a zero-volt source across the port, + to -
                    VoltageSource v{};
                    v.name = toUpper(a.name) + "#IN" + std::to_string(sense++);
                    v.nodePositive = pt.n1; v.nodeNegative = pt.n2;
                    r.a = (int)blk.voltageSources.size();
                    blk.voltageSources.push_back(v);
                }
            }
            refs.push_back(r);
        }
    }
    // conductance ports (g, gd): their voltages follow the inputs, in port order, for every output; then the
    // current of each h / hd port (a voltage source: its index among the block's voltage behavioural sources,
    // counting the outputs below in their order)
    for (size_t s = 0; s < specs.size(); ++s)
        if (specs[s].dir == 'b') for (const auto& pt : ports[s]) refs.push_back({0, pt.n1, pt.n2, ""});  // NULL: 0 - 0
    {
        int bv = 0;
        for (const auto& o : blk.behavioural) bv += o.voltage;
        for (size_t s = 0; s < specs.size(); ++s) {
            if (specs[s].dir == 'i') continue;
            for (const auto& pt : ports[s]) {
                if (pt.type == "null") continue;
                const bool v = pt.type == "v" || pt.type == "vd" || pt.type == "h" || pt.type == "hd";
                if (specs[s].dir == 'h') refs.push_back({3, bv, 0, ""});
                bv += v;
            }
        }
    }
    // outputs: a voltage (v, vd) or current (i, id; g, gd: the port's own current), each its own model copy
    // knowing which output it is (AnalogModel::out)
    int outIndex = 0;
    for (size_t s = 0; s < specs.size(); ++s) {
        if (specs[s].dir == 'i') continue;
        for (const auto& pt : ports[s]) {
            if (pt.type == "null") { ++outIndex; continue; }
            Behavioural b;
            b.name = toUpper(a.name);
            b.outPos = pt.n1; b.outNeg = pt.n2;
            b.voltage = pt.type == "v" || pt.type == "vd" || pt.type == "h" || pt.type == "hd";
            b.refs = refs;
            b.cm = m;
            if (outIndex > 0) {
                auto copy = std::make_shared<xspice::AnalogModel>(*m);
                copy->out = outIndex;
                b.cm = copy;
            }
            ++outIndex;
            if (xspice::stateful(kind)) {
                b.cmState = std::make_shared<xspice::AnalogState>();
                b.cmState->out = m->outIc;
                b.cmState->v = m->intIc;  // s_xfer: the operating point reads its initial states
            }
            blk.behavioural.push_back(b);
        }
    }
}

}  // namespace acutesim
