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
double SPICEParser::instanceArea(const std::vector<std::string>& tok, size_t from, InstTemp* t) const {
    double area = 1.0;
    for (size_t i = from; i < tok.size(); ++i) {
        const std::string k = toUpper(tok[i]);
        if (i == from && isValue(tok[i])) area *= parseValue(tok[i]);
        else if (t && (k == "TEMP" || k == "DTEMP") && i + 1 < tok.size() && isValue(tok[i + 1])) (k == "TEMP" ? t->temp : t->dtemp) = parseValue(tok[++i]);
        else if ((k == "AREA" || k == "M") && i + 1 < tok.size() && isValue(tok[i + 1])) area *= parseValue(tok[++i]);
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
    if (it != map.end()) return it->second;
    if (inSubckt_ && globals_.count(upper)) subcktGlobals_[subcktName_][next] = upper;  // see expandInstance
    map[upper] = next;
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
                                    const std::map<std::string, double>& paramOverrides) {
    result_ = SPICEParseResult{};
    tnomC_ = 27.0;
    nodeMap_.clear();
    nextNode_ = 1;
    inSubckt_ = false;
    topInstances_.clear();
    pendingICs_.clear();
    subcktICs_.clear();
    globals_.clear();
    subcktGlobals_.clear();
    for (auto& n : pzNames_) n.clear();
    sensOut_.clear();
    sensRef_.clear();
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
                    // .sens v(out[,ref]) | i(vsrc)   (AC sensitivity, .sens ... ac dec n f1 f2, is not supported yet)
                    const std::string kind = tok.size() > 2 ? toUpper(tok[1]) : "";
                    const size_t want = kind == "V" && tok.size() > 3 && toUpper(tok[3]) != "V" && toUpper(tok[3]) != "I" && toUpper(tok[3]) != "AC" ? 4 : 3;
                    if (kind != "V" && kind != "I") throw std::runtime_error(".sens expects v(out[,ref]) or i(vsrc)");
                    if (tok.size() > want && toUpper(tok[want]) == "AC") throw std::runtime_error(".sens ... ac (AC sensitivity) is not supported yet");
                    if (tok.size() != want) throw std::runtime_error(".sens supports a single v(out[,ref]) or i(vsrc) output");
                    result_.hasSens = true;
                    if (kind == "I") result_.sensSource = toUpper(tok[2]);
                    else { sensOut_ = toUpper(tok[2]); sensRef_ = want == 4 ? toUpper(tok[3]) : "0"; }
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
            mult_ = std::strchr("RLIGFJ", type) ? takeMultiplier(tok) : 1.0;

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
                case 'S': case 'W': parseSwitch(tok, blk); break;
                case 'N': parseOsdiDevice(tok, blk);  break;
                case 'B': parseBehav(line, tok, blk); break;
                case 'E': nonlinearForm(tok) ? parseBehav(line, tok, blk) : parseVCVS(tok, blk); break;
                case 'G': nonlinearForm(tok) ? parseBehav(line, tok, blk) : parseVCCS(tok, blk); break;
                case 'H': nonlinearForm(tok) ? parseBehav(line, tok, blk) : parseCCVS(tok, blk); break;
                case 'F': nonlinearForm(tok) ? parseBehav(line, tok, blk) : parseCCCS(tok, blk); break;
                case 'T': parseTLine(tok, blk);       break;
                case 'O':
                    throw std::runtime_error(tok[0] + ": lossy transmission lines (LTRA) are not supported yet");
                case 'X': {
                    // X<name> <nodes...> <subcktName>; expanded after parsing
                    requireFields(tok, 3);
                    PendingInstance pi{toUpper(tok.back()), {}, lineNum, toUpper(tok[0])};
                    for (size_t i = 1; i + 1 < tok.size(); ++i)
                        pi.nodes.push_back(nodeIndex(tok[i]));
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
    auto mosSeries = [&](TensorBlock& blk, bool top) {
        const size_t n = blk.mosfets.size();
        for (size_t k = 0; k < n; ++k) {
            Mosfet& m = blk.mosfets[k];
            const ModelCard* c;
            try { c = mosCard(m); } catch (const std::exception&) { continue; }  // resolveBlock reports it
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
            add(&JFET::drain, "RD", it->second.get("RD", 0.0));
            add(&JFET::source, "RS", it->second.get("RS", 0.0));
        }
    };
    try {
        mosSeries(result_.netlist.globalBlock, true);
        jfetSeries(result_.netlist.globalBlock, true);
        for (auto& [name, blk] : result_.netlist.blockDefinitions) { mosSeries(blk, false); jfetSeries(blk, false); }
    } catch (const std::exception& e) { return fail(lines.back().first, e.what()); }

    // Expand X cards (recursively) now that every .subckt is defined
    for (const auto& pi : topInstances_) {
        try { expandInstance(pi, pi.nodes, 0); }
        catch (const std::exception& e) { return fail(pi.line, e.what()); }
    }

    // .ic / .nodeset may name a node defined later in the file, or a subcircuit internal (x1.n)
    for (const auto& ic : pendingICs_) {
        auto it = nodeMap_.find(ic.node);
        const bool top = it != nodeMap_.end();
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
    // The card's type must fit the device (a Q on an NMOS card reported "VTO is not supported")
    auto kind = [&](const ModelCard* c, const std::string& inst, std::initializer_list<const char*> types) {
        for (const char* t : types) if (c->type == t) return c;
        throw std::runtime_error(inst + ": model " + c->name + " is " + c->type + ", not a " +
                                 std::string(*types.begin()) + "/" + *(types.begin() + 1) + " model");
    };
    auto resolveBlock = [&](TensorBlock& blk) {
        // BSIM4 (LEVEL=54 or 14) and BSIM3 (LEVEL=49 or 8) cards, ngspice's native models, run on the OSDI
        // bsim4va / bsim3_va (OpenVAF, loaded from TREESPICE_OSDI_PATH as bsim4.osdi / bsim3.osdi when needed).
        // Their geometry parameters are model parameters there, so each card plus instance geometry is one
        // OSDI model (shared by equal devices).
        auto bsimOsdi = [&](const Mosfet& m, const ModelCard& c, bool v3) {
            const char* module = v3 ? "bsim3_va" : "bsim4va";
            if (!std::isnan(m.inst.temp) || m.inst.dtemp != 0.0)
                throw std::runtime_error(m.instanceName + ": TEMP/DTEMP on a BSIM" + (v3 ? "3" : "4") + " device is not supported yet");
            if (!osdi::findModule(module)) osdi::loadLibrary(v3 ? "bsim3.osdi" : "bsim4.osdi");
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
            for (const auto& [k, v] : m.given) {
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
            if (!v3 && p.count("VERSION") && p["VERSION"] < 4.6 && (!p.count("TEMPMOD") || p["TEMPMOD"] == 0.0)) {
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
            // ponytail: ngspice also runs a negative hysteresis (it turns on at VT+VH and off at VT-VH); not modelled
            if (!(s.vh >= 0.0 && s.ron > 0.0 && s.roff > 0.0))
                throw std::runtime_error("model " + c->name + ": needs " + (w ? "IH" : "VH") + " >= 0 (negative is not supported yet) and RON, ROFF > 0");
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
            cap.capacitance_farads *= cap.m;  // was parsed and never applied
            cap.m = 1.0;
        }
        std::vector<Mosfet> level1;
        for (auto& m : blk.mosfets) {
            const ModelCard* c = kind(mosCard(m), m.instanceName, {"NMOS", "PMOS"});
            if (const double level = c->get("LEVEL", 1.0); level == 54.0 || level == 14.0 || level == 49.0 || level == 8.0) {
                blk.osdiDevices.push_back(bsimOsdi(m, *c, level == 49.0 || level == 8.0));
                continue;
            }
            const int level = (int)c->get("LEVEL", 1.0);
            const bool hi = level == 2 || level == 3;  // native levels 2 and 3 (mos2*.c, mos3*.c)
            if (level != 1 && !hi) throw std::runtime_error("model " + c->name + ": LEVEL " + std::to_string(level) + " is not supported yet");
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
            } else
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
            k.level = hi ? level : 1;
            k.tox = c->get("TOX", hi ? 1e-7 : 0.0);  // mos2set / mos3set: 100 nm unless given
            // No KP with TOX: KP = UO[cm^2/Vs] * COX (ngspice mos1temp)
            k.kp = c->get("KP", k.tox > 0.0 ? c->get("UO", c->get("U0", 600.0)) * 1e-4 * 3.9 * 8.854214871e-12 / k.tox
                                            : 2e-5);
            k.vto = c->get("VTO", 0.0); k.lambda = c->get("LAMBDA", 0.0); k.gamma = c->get("GAMMA", 0.0);
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
                if (!c->params.count("VTO")) {
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
            if (!(m.l - 2.0 * k.ld + k.xl > 0.0 && m.w - 2.0 * k.wd + k.xw > 0.0))
                throw std::runtime_error(m.instanceName + ": effective L (L - 2*LD + XL) and W (W - 2*WD + XW) must be > 0");
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
            supported(c, {"IS", "N", "RS", "CJO", "CJ0", "CJ", "VJ", "PB", "M", "MJ", "FC", "TT", "BV", "IBV",
                          "EG", "XTI", "TNOM", "TM1", "TM2", "TCV", "TTT1", "TTT2", "TRS", "TRS1", "TRS2",
                          "JSW", "NS", "CJSW", "MJSW", "PHP", "IKF", "IK", "IKR", "KF", "AF"});
            bounded(c, {"IS", "RS", "CJO", "CJ0", "CJ", "M", "MJ", "TT", "JSW", "CJSW", "MJSW", "IKF", "IK", "IKR", "KF"}, {"N", "BV", "NS", "PHP"});  // ngspice: BV = 0 breaks down at 0 V
            d.saturationCurrent_I_S_A = c->get("IS", 1e-14);
            d.emissionCoefficient_N   = c->get("N", 1.0);
            auto& k = d.card;
            k.rs = c->get("RS", 0.0); k.cjo = first(c, {"CJO", "CJ0", "CJ"}, 0.0);
            k.vj = first(c, {"VJ", "PB"}, 1.0); k.m = first(c, {"M", "MJ"}, 0.5);
            k.fc = c->get("FC", 0.5); k.tt = c->get("TT", 0.0);
            k.jsw = c->get("JSW", 0.0); k.ns = c->get("NS", 1.0); k.cjsw = c->get("CJSW", 0.0);
            k.mjsw = c->get("MJSW", 0.33); k.php = c->get("PHP", 1.0);
            k.ikf = first(c, {"IKF", "IK"}, 0.0); k.ikr = c->get("IKR", 0.0);
            k.kf = c->get("KF", 0.0); k.af = c->get("AF", 1.0);
            k.bvGiven = c->params.count("BV") > 0;  // BV > 0 (checked above)
            k.bv = c->get("BV", 0.0); k.ibv = c->get("IBV", 1e-3);
            if (!(k.m < 1.0 && k.mjsw < 1.0 && k.vj > 0.0 && k.fc < 1.0 && k.rs >= 0.0 && k.ibv > 0.0))
                throw std::runtime_error("model " + c->name + ": needs M, MJSW < 1, VJ > 0, FC < 1, RS >= 0, IBV > 0");
            using namespace devtemp;
            const double Tn = c->get("TNOM", tnomC_) + 273.15, T = instK(d.inst), dt = T - Tn;
            d.thermalVoltage_V_T_V = vt(T);
            k.m = std::min(k.m * poly(c->get("TM1", 0.0), c->get("TM2", 0.0), dt), 0.9);
            if (T != Tn) {
                d.saturationCurrent_I_S_A *= isFactor(T, Tn, std::max(c->get("EG", 1.11), 0.1), c->get("XTI", 3.0), d.emissionCoefficient_N);
                k.cjo *= junctionAt(k.vj, k.m, Tn, T);
                k.jsw *= isFactor(T, Tn, std::max(c->get("EG", 1.11), 0.1), c->get("XTI", 3.0), d.emissionCoefficient_N);
                k.cjsw *= junctionAt(k.php, k.mjsw, Tn, T);
            }
            if (k.fc * k.vj > 1.0) k.vj = 1.0 / k.fc;  // diotemp: the depletion knee stays below 1 V
            k.tt *= poly(c->get("TTT1", 0.0), c->get("TTT2", 0.0), dt);
            k.rs *= poly(first(c, {"TRS1", "TRS"}, 0.0), c->get("TRS2", 0.0), dt);
            if (k.bvGiven) k.bv -= c->get("TCV", 0.0) * dt;
        }
        for (auto& q : blk.bjts) if (const ModelCard* c = kind(card(q.modelName), q.instanceName, {"NPN", "PNP"})) {
            supported(c, {"IS", "BF", "BR", "NF", "NR", "VAF", "VA", "VAR", "VB", "IKF", "IK", "IKR",
                          "ISE", "NE", "ISC", "NC", "RB", "RC", "RE", "CJE", "VJE", "PE", "MJE", "ME",
                          "CJC", "VJC", "PC", "MJC", "MC", "XCJC", "CJS", "CCS", "VJS", "PS", "MJS", "MS",
                          "FC", "TF", "XTF", "VTF", "ITF", "TR", "EG", "XTI", "XTB", "TNOM", "RBM", "IRB", "NK", "KF", "AF", "ISS", "NS", "SUBS"});
            bounded(c, {"IS", "VAF", "VA", "VAR", "VB", "IKF", "IK", "IKR", "ISE", "ISC", "CJE", "CJC", "CJS", "CCS",
                        "MJE", "ME", "MJC", "MC", "MJS", "MS", "TF", "XTF", "VTF", "ITF", "TR", "ISS", "RBM", "IRB", "NK", "KF"},
                    {"NE", "NC", "NS"});
            q.isNPN = (c->type != "PNP");
            q.saturationCurrent_I_S_A = c->get("IS", 1e-16);
            q.betaF = c->get("BF", 100.0);
            q.betaR = c->get("BR", 1.0);
            auto& k = q.card;
            k.nf = c->get("NF", 1.0); k.nr = c->get("NR", 1.0);
            k.vaf = first(c, {"VAF", "VA"}, 0.0); k.var = first(c, {"VAR", "VB"}, 0.0);
            k.ikf = first(c, {"IKF", "IK"}, 0.0); k.ikr = c->get("IKR", 0.0);
            k.ise = c->get("ISE", 0.0); k.ne = c->get("NE", 1.5);
            k.isc = c->get("ISC", 0.0); k.nc = c->get("NC", 2.0);
            k.rb = c->get("RB", 0.0); k.rc = c->get("RC", 0.0); k.re = c->get("RE", 0.0);
            k.cje = c->get("CJE", 0.0); k.vje = first(c, {"VJE", "PE"}, 0.75); k.mje = first(c, {"MJE", "ME"}, 0.33);
            k.cjc = c->get("CJC", 0.0); k.vjc = first(c, {"VJC", "PC"}, 0.75); k.mjc = first(c, {"MJC", "MC"}, 0.33);
            k.xcjc = c->get("XCJC", 1.0);
            k.cjs = first(c, {"CJS", "CCS"}, 0.0); k.vjs = first(c, {"VJS", "PS"}, 0.75); k.mjs = first(c, {"MJS", "MS"}, 0.0);
            k.fc = c->get("FC", 0.5);
            k.iss = c->get("ISS", 0.0); k.ns = c->get("NS", 1.0);
            // SUBS (ngspice): 1 vertical (substrate junction to C), -1 lateral (to B); default by polarity
            const double subs = c->get("SUBS", q.isNPN ? 1.0 : -1.0);
            if (subs != 1.0 && subs != -1.0) throw std::runtime_error("model " + c->name + ": SUBS must be 1 or -1");
            k.lateral = subs == -1.0;
            k.tf = c->get("TF", 0.0); k.xtf = c->get("XTF", 0.0); k.vtf = c->get("VTF", 0.0);
            k.itf = c->get("ITF", 0.0); k.tr = c->get("TR", 0.0);
            k.rbm = c->get("RBM", k.rb); k.irb = c->get("IRB", 0.0); k.nk = c->get("NK", 0.5);
            k.kf = c->get("KF", 0.0); k.af = c->get("AF", 1.0);
            const double Tn = c->get("TNOM", tnomC_) + 273.15, T = instK(q.inst);
            q.thermalVoltage_V_T_V = devtemp::vt(T);
            if (T != Tn) {
                using namespace devtemp;
                const double ratlog = std::log(T / Tn), factlog = (T / Tn - 1) * c->get("EG", 1.11) / vt(T) + c->get("XTI", 3.0) * ratlog;
                const double fac = std::exp(factlog), bf = std::exp(ratlog * c->get("XTB", 0.0));
                q.saturationCurrent_I_S_A *= fac; k.iss *= fac;
                q.betaF *= bf; q.betaR *= bf;
                k.ise *= std::exp(factlog / k.ne) / bf; k.isc *= std::exp(factlog / k.nc) / bf;
                k.cje *= junctionAt(k.vje, k.mje, Tn, T);
                k.cjc *= junctionAt(k.vjc, k.mjc, Tn, T);
                k.cjs *= junctionAt(k.vjs, k.mjs, Tn, T);
            }
            if (!(k.mje < 1.0 && k.mjc < 1.0 && k.mjs < 1.0 && k.vje > 0.0 && k.vjc > 0.0 && k.vjs > 0.0 &&
                  k.fc < 1.0 && k.xcjc >= 0.0 && k.xcjc <= 1.0 && k.nf > 0.0 && k.nr > 0.0 &&
                  k.rb >= 0.0 && k.rc >= 0.0 && k.re >= 0.0 && q.betaF > 0.0 && q.betaR > 0.0))
                throw std::runtime_error("model " + c->name + ": needs MJE/MJC/MJS < 1, VJE/VJC/VJS > 0, FC < 1, "
                                         "0 <= XCJC <= 1, NF/NR/BF/BR > 0, RB/RC/RE >= 0");
        }
        for (auto& o : blk.osdiDevices) {
            if (o.model) continue;  // a BSIM4 card, bridged above
            const ModelCard* c = card(o.modelName);
            const osdi::Descriptor* d = osdi::findModule(c->type);
            if (!d) throw std::runtime_error("model " + c->name + ": type " + c->type +
                                             " is not a loaded OSDI module (load it with .osdi <file>)");
            auto& m = osdiModels_[toUpper(c->name)];
            if (!m) m = std::make_shared<osdi::Model>(d, c->name, c->params);
            o.model = m;
        }
        for (auto& j : blk.jfets) if (const ModelCard* c = card(j.modelName)) {
            if (c->type != "NJF" && c->type != "PJF" && c->type != "NJFET" && c->type != "PJFET")
                throw std::runtime_error(j.instanceName + ": model " + c->name + " is " + c->type + ", not an NJF/PJF model");
            supported(c, {"BETA", "VTO", "LAMBDA", "IS", "N", "B", "CGS", "CGD", "PB", "FC", "KF", "AF", "RD", "RS",
                          "TNOM", "XTI", "EG", "VTOTC", "BETATCE", "TCV", "BEX"});
            bounded(c, {"BETA", "LAMBDA", "IS", "CGS", "CGD", "KF", "RD", "RS", "XTI"}, {"N", "B", "PB", "EG"});
            // ngspice jfettemp: IS, PB and CGS/CGD by the junction laws; VTO by VTOTC (else TCV), BETA by BETATCE (else BEX)
            const double T = ambientK_, Tn = c->get("TNOM", tnomC_) + 273.15, dT = T - Tn;
            j.thermalVoltage_V_T_V = devtemp::vt(T);
            j.isNChannel = (c->type == "NJF" || c->type == "NJFET");
            j.beta   = c->get("BETA", 1e-4) * j.m * (c->params.count("BETATCE") ? std::pow(1.01, c->get("BETATCE", 0.0) * dT)
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
                if (s.type == "EXP") {
                    auto& w = s.wf;
                    auto dflt = [](double& x, double v) { if (std::isnan(x)) x = v; };
                    if (s.type == "EXP") { dflt(w[2], ts); dflt(w[3], ts); dflt(w[4], w[2] + ts); dflt(w[5], ts); }
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
    try {  // OSDI instances: parameters, setup_instance, node collapsing
        result_.netlist.instantiateOsdi(result_.netlist.environment.ambient_temp_K);
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
    size_t i = 3;
    const char c0 = tok[3][0];
    const bool valueFirst = (tok.size() - 3) % 2 == 0 || std::isdigit((unsigned char)c0) || c0 == '.' || c0 == '-' || c0 == '+';
    c.capacitance_farads = valueFirst ? parseValue(tok[i++]) : NAN;  // NAN: from the model (resolveBlock)
    if ((tok.size() - i) % 2 == 1) c.model_name = tok[i++];

    // tokenize() turns "KEY=val" into ["KEY", "val"]. IC= applies with UIC.
    for (; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (i + 1 >= tok.size()) throw std::runtime_error(tok[0] + ": parameter " + upper + " has no value");
        if (upper == "W")      c.width_m  = parseValue(tok[++i]);
        else if (upper == "L") c.length_m = parseValue(tok[++i]);
        else if (upper == "M") c.m        = parseValue(tok[++i]);
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
    ind.inductance_henries = parseValue(tok[3]) / mult_;
    // IC= (initial current) applies with UIC; anything else used to be ignored silently
    for (size_t i = 4; i < tok.size(); ++i) {
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
            const size_t lo = kw == "AM" ? 5 : kw == "SFFM" ? 5 : 2, hi = kw == "EXP" || kw == "AM" ? 6 : 5;
            const double nan = std::nan("");
            s.wf = kw == "EXP" ? std::vector<double>{0, 0, nan, nan, nan, nan}
                 : std::vector<double>(kw == "SFFM" ? 5 : 6, 0.0);
            size_t n = 0;
            for (++i; i < tok.size() && n < hi && isValue(tok[i]); ++i, ++n) s.wf[n] = parseValue(tok[i]);
            if (n < lo) throw std::runtime_error(tok[0] + ": " + kw + " needs at least " + std::to_string(lo) + " values");
            if (i < tok.size()) throw std::runtime_error(tok[0] + ": " + kw + " takes at most " + std::to_string(hi) + " values");
            if (kw == "EXP")  // ngspice: a 0 TD1, TAU1, TD2 or TAU2 is "not given"
                for (size_t k = 2; k < 6; ++k) if (s.wf[k] == 0.0) s.wf[k] = nan;
            // ngspice limits the deviation to FC: the index is min(MDI, FC/FS). Its FC / FS defaults
            // (1/TSTOP) did not match the documented sine, so they are not offered.
            if (kw == "SFFM") {
                if (!(s.wf[2] > 0.0 && s.wf[4] > 0.0 && s.wf[3] >= 0.0))
                    throw std::runtime_error(tok[0] + ": SFFM needs FC > 0, MDI >= 0 and FS > 0 (defaults are not supported yet)");
                s.wf[3] = std::min(s.wf[3], s.wf[2] / s.wf[4]);
            }
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
void SPICEParser::parseDiode(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 4);
    Diode d;
    std::vector<std::string> rest(tok.begin(), tok.begin() + 4);  // PJ= is the sidewall perimeter; M= also scales it
    for (size_t i = 4; i < tok.size(); ++i) {
        const std::string k = toUpper(tok[i]);
        if (k == "PJ" && i + 1 < tok.size() && isValue(tok[i + 1])) d.pj = parseValue(tok[++i]);
        else { if (k == "M" && i + 1 < tok.size() && isValue(tok[i + 1])) d.mult = parseValue(tok[i + 1]); rest.push_back(tok[i]); }
    }
    d.area = instanceArea(rest, 4, &d.inst);
    d.instanceName = tok[0];
    d.anode = nodeIndex(tok[1]);
    d.cathode = nodeIndex(tok[2]);
    d.modelName = tok[3];
    blk.diodes.push_back(d);
}

// M<name> <nd> <ng> <ns> <nb> <modelName> [W=] [L=] [AD=] [AS=] [PD=] [PS=] [NRD=] [NRS=] [NF=] [M=]
void SPICEParser::parseMosfet(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 6);
    Mosfet m;
    m.instanceName = tok[0];
    m.drain  = nodeIndex(tok[1]);
    m.gate   = nodeIndex(tok[2]);
    m.source = nodeIndex(tok[3]);
    m.body   = nodeIndex(tok[4]);
    m.modelName = tok[5];
    const std::string mu = toUpper(tok[5]);  // a .model card, or a binned family NAME.0, NAME.1, ...
    const auto bin = modelNames_.lower_bound(mu + ".");
    if (!modelNames_.count(mu) && (bin == modelNames_.end() || bin->compare(0, mu.size() + 1, mu + ".") != 0))
        throw std::runtime_error(tok[0] + ": expected 4 nodes (d g s b), then a model name; '" + tok[5] +
                                 "' is not a .model card");
    m.w = m.l = 100e-6;  // SPICE DEFW / DEFL (not scaled)

    // tokenize() turns "W=4u" into ["W", "4u"]. AD/AS/PD/PS act through CJ/CJSW;
    // NRD/NRS only through RSH, which the model check rejects. SA/SB/SD and NF
    // reach BSIM4 (level 54); level 1 rejects NF != 1 when its card is resolved.
    // .option scale multiplies the lengths (areas by scale^2), as in ngspice.
    for (size_t i = 6; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (upper == "OFF") throw std::runtime_error(tok[0] + ": OFF is not supported yet");
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
        else throw std::runtime_error(tok[0] + ": parameter " + upper + " is not supported yet");
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

    // Optional substrate node: tok[4] is the substrate when tok[5] names a .model
    // card (all names are known after the first pass). No substrate node: ground.
    const size_t modelIdx = tok.size() >= 6 && modelNames_.count(toUpper(tok[5])) ? 5 : 4;
    if (modelIdx == 5) q.substrate = nodeIndex(tok[4]);
    q.modelName = tok[modelIdx];
    q.area = instanceArea(tok, modelIdx + 1, &q.inst);
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
    j.area = instanceArea(tok, 5);
    j.instanceName = tok[0];
    j.drain  = nodeIndex(tok[1]);
    j.gate   = nodeIndex(tok[2]);
    j.source = nodeIndex(tok[3]);
    j.modelName = tok[4];
    j.m = mult_;
    blk.jfets.push_back(j);
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

    // Parse optional parameters (key=value pairs flattened by tokenizer)
    for (size_t i = 3; i + 1 < tok.size(); i += 2) {
        std::string key = toUpper(tok[i]);
        double val = parseValue(tok[i + 1]);
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
        subcktBlock_.portNodeIndices.push_back(nodeIndex(tok[i]));  // local 1..P
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
                                 int depth, const std::string& path) {
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

    for (const auto& child : subcktChildren_[pi.subckt]) {
        std::vector<int> childPorts;
        for (int l : child.nodes) childPorts.push_back(l == 0 ? 0 : map[l - 1]);
        expandInstance(child, childPorts, depth + 1, full + ".");
    }
}

// .PARAM <name>=<value> [<name>=<value>]... (numparam::expand's evaluated globals)
void SPICEParser::parseParam(const std::vector<std::string>& tok) {
    for (size_t i = 1; i + 1 < tok.size(); i += 2) params_[toUpper(tok[i])] = parseValue(tok[i + 1]);
}

} // namespace acutesim
