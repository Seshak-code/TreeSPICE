#include "contraction_solver.h"
#include "gpu_front_lu.h"
#include "worker_pool.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <string>

namespace {

using Adj = std::vector<std::vector<int>>;

// MC64-style maximum-product matching (Duff & Koster, successive shortest
// paths with Dijkstra and dual potentials). Returns rowOfVar: the equation (A
// row) paired with each variable so that the permuted diagonal is large.
// Pivots are only searched inside a front, so without this an E source — whose
// 1e10 gain entry sits OFF the diagonal — forces pivoting on a tiny entry when
// its output equation lives in an ancestor front. Falls back to identity if A
// is structurally singular (the factorization then reports the singularity).
std::vector<int> matchLargeDiagonal(const Csr_matrix& A) {
    const int n = A.rows;
    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 0);
    // Column-major edges with cost log(colmax_j) - log|a_ij| >= 0
    std::vector<int> cp(n + 1, 0), ri;
    std::vector<double> cost;
    for (int i = 0; i < n; ++i)
        for (int k = A.row_pointer[i]; k < A.row_pointer[i + 1]; ++k)
            if (A.values[k] != 0.0) cp[A.col_indices[k] + 1]++;
    for (int j = 0; j < n; ++j) cp[j + 1] += cp[j];
    ri.resize(cp[n]);
    cost.resize(cp[n]);
    std::vector<double> colmax(n, 0.0);
    {
        std::vector<int> fill(cp.begin(), cp.end() - 1);
        for (int i = 0; i < n; ++i)
            for (int k = A.row_pointer[i]; k < A.row_pointer[i + 1]; ++k) {
                if (A.values[k] == 0.0) continue;
                int j = A.col_indices[k];
                ri[fill[j]] = i;
                cost[fill[j]++] = std::abs(A.values[k]);
                colmax[j] = std::max(colmax[j], std::abs(A.values[k]));
            }
    }
    for (int j = 0; j < n; ++j) {
        if (cp[j] == cp[j + 1]) return identity;
        for (int t = cp[j]; t < cp[j + 1]; ++t) cost[t] = std::log(colmax[j]) - std::log(cost[t]);
    }

    const double INF = std::numeric_limits<double>::infinity();
    std::vector<int> rowMatch(n, -1), colMatch(n, -1), pred(n, -1), touched, finalized;
    std::vector<double> pr(n, 0.0), pc(n, 0.0), dist(n, INF);
    std::vector<char> done(n, 0);
    std::vector<std::pair<int, double>> colsReached;
    using QE = std::pair<double, int>;
    for (int j0 = 0; j0 < n; ++j0) {
        std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
        touched.clear(); finalized.clear(); colsReached.assign(1, {j0, 0.0});
        auto relax = [&](int j, double dj) {
            for (int t = cp[j]; t < cp[j + 1]; ++t) {
                int i = ri[t];
                if (done[i]) continue;
                double nd = dj + cost[t] + pc[j] - pr[i];  // reduced cost >= 0
                if (nd < dist[i]) {
                    if (dist[i] == INF) touched.push_back(i);
                    dist[i] = nd; pred[i] = j; pq.push({nd, i});
                }
            }
        };
        relax(j0, 0.0);
        int freeRow = -1;
        double D = 0.0;
        while (!pq.empty()) {
            auto [d, i] = pq.top(); pq.pop();
            if (done[i] || d > dist[i]) continue;
            done[i] = 1; finalized.push_back(i);
            if (rowMatch[i] < 0) { freeRow = i; D = d; break; }
            colsReached.push_back({rowMatch[i], d});
            relax(rowMatch[i], d);
        }
        if (freeRow < 0) return identity;  // structurally singular
        for (int i : finalized) pr[i] += dist[i] - D;   // Johnson reweighting,
        for (auto& [j, dj] : colsReached) pc[j] += dj - D;  // shifted by -D
        for (int i = freeRow;;) {  // flip the augmenting path
            int j = pred[i], prev = colMatch[j];
            colMatch[j] = i; rowMatch[i] = j;
            if (j == j0) break;
            i = prev;
        }
        for (int t : touched) { dist[t] = INF; done[t] = 0; }
    }
    return colMatch;  // colMatch[j] = row paired with variable j
}

// Undirected graph of the row-permuted pattern B = P*A (B's row v is A's row
// rowOfVar[v]), symmetrised, no self loops.
Adj symmetricAdjacency(const Csr_matrix& A, const std::vector<int>& rowOfVar) {
    Adj adj(A.rows);
    for (int v = 0; v < A.rows; ++v)
        for (int k = A.row_pointer[rowOfVar[v]]; k < A.row_pointer[rowOfVar[v] + 1]; ++k) {
            int j = A.col_indices[k];
            if (j == v) continue;
            adj[v].push_back(j);
            adj[j].push_back(v);
        }
    for (auto& a : adj) {
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }
    return adj;
}

// Nested dissection producing fronts in pre-order (parents before children).
// ponytail: separators are BFS level sets from a pseudo-peripheral vertex —
// cheap and exact as vertex separators, but larger than METIS-quality ones;
// a multilevel partitioner is the upgrade path once fronts dominate runtime.
struct Dissector {
    const Adj& adj;
    int leafSize;
    std::vector<int> region;  // region label of each vertex; -1 = already placed in a front
    int nextRegion = 0;
    std::vector<ContractionFront> fronts;

    int addFront(std::vector<int> pivots, int parent) {
        ContractionFront f;
        f.pivots = std::move(pivots);
        f.parent = parent;
        for (int v : f.pivots) region[v] = -1;
        fronts.push_back(std::move(f));
        return (int)fronts.size() - 1;
    }

    // BFS restricted to vertices labelled `rid`; returns level sets.
    std::vector<std::vector<int>> levelsFrom(int start, int rid) {
        std::vector<std::vector<int>> levels{{start}};
        std::vector<int> seen;  // vertices visited, to restore labels
        region[start] = -2 - rid;  // temporary "visited" label
        seen.push_back(start);
        while (true) {
            std::vector<int> next;
            for (int v : levels.back())
                for (int w : adj[v])
                    if (region[w] == rid) {
                        region[w] = -2 - rid;
                        seen.push_back(w);
                        next.push_back(w);
                    }
            if (next.empty()) break;
            std::sort(next.begin(), next.end());  // determinism
            levels.push_back(std::move(next));
        }
        for (int v : seen) region[v] = rid;
        return levels;
    }

    void dissect(std::vector<int> S, int parent) {
        if (S.empty()) return;
        if ((int)S.size() <= leafSize) { addFront(std::move(S), parent); return; }

        int rid = nextRegion++;
        for (int v : S) region[v] = rid;

        // Split into connected components first (independent subtrees)
        std::vector<std::vector<int>> comps;
        for (int v : S) {
            if (region[v] != rid) continue;
            auto lv = levelsFrom(v, rid);
            std::vector<int> comp;
            for (auto& l : lv) comp.insert(comp.end(), l.begin(), l.end());
            for (int u : comp) region[u] = nextRegion;  // retire from this pass
            ++nextRegion;
            comps.push_back(std::move(comp));
        }
        if (comps.size() > 1) {
            for (auto& c : comps) dissect(std::move(c), parent);
            return;
        }

        // Single component: vertex separator = the smallest balanced BFS level
        // of a level structure rooted at a pseudo-peripheral vertex.
        rid = nextRegion++;
        for (int v : S) region[v] = rid;
        auto lv = levelsFrom(S.front(), rid);
        lv = levelsFrom(lv.back().front(), rid);  // second sweep: pseudo-peripheral start
        int d = (int)lv.size();
        if (d < 3) { addFront(std::move(S), parent); return; }  // no useful separator

        const int total = (int)S.size();
        int best = -1, before = 0;
        for (int m = 0; m < d; ++m) {
            int after = total - before - (int)lv[m].size();
            bool balanced = m > 0 && m < d - 1 && before >= total / 4 && after >= total / 4;
            if (balanced && (best < 0 || lv[m].size() < lv[best].size())) best = m;
            before += (int)lv[m].size();
        }
        if (best < 0) {  // no level in the balanced window: take the median level
            before = 0;
            for (int m = 1; m < d - 1; ++m) {
                before += (int)lv[m - 1].size();
                if (before >= total / 2) { best = m; break; }
            }
            if (best < 0) best = d / 2;
        }
        if ((int)lv[best].size() * 2 > total) { addFront(std::move(S), parent); return; }

        std::vector<int> left, right;
        for (int m = 0; m < best; ++m) left.insert(left.end(), lv[m].begin(), lv[m].end());
        for (int m = best + 1; m < d; ++m) right.insert(right.end(), lv[m].begin(), lv[m].end());
        int f = addFront(lv[best], parent);
        dissect(std::move(left), f);
        dissect(std::move(right), f);
    }
};

std::vector<int> sortedUnion(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out;
    out.reserve(a.size() + b.size());
    std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return out;
}

// Approximate minimum degree (Amestoy, Davis & Duff) on the quotient graph of the
// vertices in `vars`: eliminating a vertex p turns it into an element whose list
// L[p] is p's elimination-graph neighbourhood; elements it touches are absorbed
// into it (they become its children in the assembly tree), and so is any element
// whose list the new one covers (aggressive absorption). Degrees are AMD's bound
// |A_i| + |L_p| - 1 + sum over other elements e of |L_e \ L_p|. Fronts are the
// fundamental supernodes of the assembly tree (a single child whose list is the
// parent's plus the parent): what KLU's AMD gives sparse circuit matrices, where
// BFS nested dissection builds huge dense separators (up to 20 000x KLU's flops).
// ponytail: no supervariables or mass elimination; they speed up the ordering of
// mesh-like graphs, not its quality on circuits.
std::vector<ContractionFront> minDegreeFronts(const Adj& adj, const std::vector<int>& vars) {
    const int n = (int)adj.size();
    enum : char { OUT, LIVE, ELEMENT, ABSORBED };
    std::vector<char> state(n, OUT);
    for (int v : vars) state[v] = LIVE;
    Adj A(n), E(n), L(n);
    std::vector<int> deg(n, 0), parent(n, -1), w(n, -1), mark(n, -1), cnt(n, 0);  // cnt: |L_p|
    // Degree buckets (doubly linked lists, as AMD): O(1) insert / remove / min
    std::vector<int> head(n + 1, -1), nxt(n, -1), prv(n, -1);
    int minDeg = n;
    auto bucketIn = [&](int v) {
        nxt[v] = head[deg[v]]; prv[v] = -1;
        if (head[deg[v]] >= 0) prv[head[deg[v]]] = v;
        head[deg[v]] = v;
        minDeg = std::min(minDeg, deg[v]);
    };
    auto bucketOut = [&](int v) {
        if (prv[v] >= 0) nxt[prv[v]] = nxt[v]; else head[deg[v]] = nxt[v];
        if (nxt[v] >= 0) prv[nxt[v]] = prv[v];
    };
    for (int v : vars) {
        for (int u : adj[v]) if (state[u] == LIVE) A[v].push_back(u);
        deg[v] = (int)A[v].size();
        bucketIn(v);
    }
    std::vector<int> order;
    order.reserve(vars.size());
    int live = (int)vars.size();
    while (live > 0) {
        while (head[minDeg] < 0) ++minDeg;
        const int p = head[minDeg];
        bucketOut(p);
        order.push_back(p);
        state[p] = ELEMENT;
        --live;
        // L_p: live neighbours through variables and through absorbed elements
        auto& Lp = L[p];
        mark[p] = p;
        for (int u : A[p]) if (state[u] == LIVE && mark[u] != p) { mark[u] = p; Lp.push_back(u); }
        for (int e : E[p]) {
            if (state[e] != ELEMENT) continue;
            for (int u : L[e]) if (state[u] == LIVE && mark[u] != p) { mark[u] = p; Lp.push_back(u); }
            state[e] = ABSORBED;
            parent[e] = p;
            Adj::value_type().swap(L[e]);
        }
        cnt[p] = (int)Lp.size();
        Adj::value_type().swap(A[p]);
        Adj::value_type().swap(E[p]);
        // w[e] = |L_e \ L_p| for the other elements next to L_p
        for (int i : Lp)
            for (int e : E[i]) {
                if (state[e] != ELEMENT) continue;
                if (w[e] < 0) w[e] = (int)L[e].size();
                --w[e];
            }
        for (int i : Lp) {
            auto& Ei = E[i];
            int d = (int)Lp.size() - 1;
            size_t k = 0;
            for (int e : Ei) {
                if (state[e] != ELEMENT) continue;
                if (w[e] == 0) { state[e] = ABSORBED; parent[e] = p; continue; }  // L_e inside L_p
                d += w[e];
                Ei[k++] = e;
            }
            Ei.resize(k);
            Ei.push_back(p);
            auto& Ai = A[i];  // edges now covered by element p go
            k = 0;
            for (int u : Ai) if (state[u] == LIVE && mark[u] != p) Ai[k++] = u;
            Ai.resize(k);
            d += (int)k;
            d = std::min(d, live - 1);
            bucketOut(i);
            deg[i] = d;
            bucketIn(i);
        }
        for (int i : Lp) for (int e : E[i]) w[e] = -1;
    }
    // Fundamental supernodes: p joins its parent q when it is q's only child and
    // |L_p| = |L_q| + 1 (same column structure below q).
    std::vector<int> children(n, 0), front(n, -1);
    for (int p : order) if (parent[p] >= 0) ++children[parent[p]];
    std::vector<ContractionFront> fronts;
    for (int p : order) {
        if (front[p] < 0) { front[p] = (int)fronts.size(); fronts.emplace_back(); }
        fronts[front[p]].pivots.push_back(p);
        const int q = parent[p];
        if (q >= 0 && children[q] == 1 && cnt[p] == cnt[q] + 1) front[q] = front[p];
    }
    for (int p : order)
        if (const int q = parent[p]; q >= 0 && front[q] != front[p]) fronts[front[p]].parent = front[q];

    // Relaxed amalgamation (CHOLMOD's default rule): a child front joins its parent when
    // the explicit zeros that adds stay small, so very sparse circuits do not pay a
    // front's fixed cost per unknown (hcircuit: 76k fronts for 2.6 Mflop). Fronts are
    // in elimination order, children before parents.
    const int nf = (int)fronts.size();
    std::vector<int> into(nf, -1), np(nf), bnd(nf);
    std::vector<double> zeros(nf, 0.0);  // explicit zeros a front already carries from merges
    for (int f = 0; f < nf; ++f) { np[f] = (int)fronts[f].pivots.size(); bnd[f] = cnt[fronts[f].pivots.back()]; }
    auto find = [&](int f) { while (into[f] >= 0) f = into[f]; return f; };
    for (int c = 0; c < nf; ++c) {
        if (fronts[c].parent < 0) continue;
        const int f = find(fronts[c].parent);
        const double cols = np[c] + np[f], rows = cols + bnd[f];
        // the child's columns padded to the parent's rows, plus what both carry already
        const double total = zeros[c] + zeros[f] + (double)np[c] * (np[f] + bnd[f] - bnd[c]);
        const double z = total / (cols * rows);
        if (cols <= 4 || (cols <= 16 && z < 0.8) || (cols <= 48 && z < 0.1) || z < 0.05) {
            into[c] = f;
            np[f] += np[c];
            zeros[f] = total;
        }
    }
    std::vector<ContractionFront> merged;
    std::vector<int> id(nf, -1);
    for (int f = 0; f < nf; ++f) if (into[f] < 0) { id[f] = (int)merged.size(); merged.emplace_back(); }
    for (int f = 0; f < nf; ++f) {
        auto& m = merged[id[find(f)]];
        m.pivots.insert(m.pivots.end(), fronts[f].pivots.begin(), fronts[f].pivots.end());
        if (into[f] < 0) m.parent = fronts[f].parent < 0 ? -1 : id[find(fronts[f].parent)];
    }
    return merged;
}

}  // namespace

ContractionPlan planContraction(const Csr_matrix& A, int leafSize) {
    ContractionPlan base;
    base.n = A.rows;
    base.rowPtr = A.row_pointer;
    base.colIdx = A.col_indices;
    const int n = A.rows;
    if (n == 0) return base;

    base.rowOfVar = matchLargeDiagonal(A);
    std::vector<int> varOfRow(n);
    for (int v = 0; v < n; ++v) varOfRow[base.rowOfVar[v]] = v;
    Adj adj = symmetricAdjacency(A, base.rowOfVar);

    // Hub nets (supply rails touching thousands of devices) go to a root front,
    // eliminated last — as AMD does with dense rows — so they never wreck the
    // ordering of the rest, which goes under it.
    const double hubDegree = std::max(16.0, 10.0 * std::sqrt((double)n));
    std::vector<int> hubs, rest;
    for (int v = 0; v < n; ++v) ((double)adj[v].size() > hubDegree ? hubs : rest).push_back(v);

    // The rest is ordered both ways and the cheaper plan is kept: nested dissection
    // wins on grids / power-delivery meshes, minimum degree on transistor-level
    // circuits. Cost = flops + 1e4 per front: a front's fixed cost (layout,
    // assembly, its vectors) measured ~0.8 us against ~1e-10 s per flop (Apple M4
    // Pro, threaded), fitted on circuit_1, where flops alone chose the 2.7x slower
    // plan. TREESPICE_ORDERING=nd|md forces one.
    auto finish = [&](std::vector<ContractionFront> pre) {
    ContractionPlan plan = base;
    std::vector<std::vector<int>> kids(pre.size());
    std::vector<int> roots;
    for (int f = 0; f < (int)pre.size(); ++f)
        (pre[f].parent < 0 ? roots : kids[pre[f].parent]).push_back(f);
    std::vector<int> order;  // pre-index in postorder
    for (int r : roots) {
        std::vector<std::pair<int, size_t>> st{{r, 0}};
        while (!st.empty()) {
            auto& [f, i] = st.back();
            if (i < kids[f].size()) { int c = kids[f][i++]; st.push_back({c, 0}); }
            else { order.push_back(f); st.pop_back(); }
        }
    }
    std::vector<int> newId(pre.size());
    for (int i = 0; i < (int)order.size(); ++i) newId[order[i]] = i;
    plan.fronts.resize(pre.size());
    for (int i = 0; i < (int)order.size(); ++i) {
        ContractionFront f = std::move(pre[order[i]]);
        f.parent = f.parent < 0 ? -1 : newId[f.parent];
        plan.fronts[i] = std::move(f);
    }
    for (int i = 0; i < (int)plan.fronts.size(); ++i)
        if (plan.fronts[i].parent >= 0) plan.fronts[plan.fronts[i].parent].children.push_back(i);

    // Elimination rank of every variable (postorder, pivots in order)
    std::vector<int> rank(n, -1);
    int r = 0;
    for (auto& f : plan.fronts) {
        std::sort(f.pivots.begin(), f.pivots.end());
        for (int v : f.pivots) rank[v] = r++;
    }

    // Symbolic phase: boundary = variables eliminated later that this front's
    // pivots touch, plus the children's boundaries (minus its own pivots).
    for (auto& f : plan.fronts) {
        const int last = rank[f.pivots.back()];
        std::vector<int> b;
        for (int v : f.pivots)
            for (int u : adj[v]) if (rank[u] > last) b.push_back(u);
        std::sort(b.begin(), b.end());
        b.erase(std::unique(b.begin(), b.end()), b.end());
        for (int c : f.children) {
            std::vector<int> cb;
            for (int u : plan.fronts[c].boundary) if (rank[u] > last) cb.push_back(u);
            b = sortedUnion(b, cb);
        }
        f.boundary = std::move(b);
    }

    // Numeric-phase maps. Entry a_ij is assembled into the front of whichever
    // of i, j is eliminated first (row pass for i, column pass for j).
    std::vector<int> colPtr(n + 1, 0), colRow, colK;
    for (int k = 0; k < (int)A.col_indices.size(); ++k) colPtr[A.col_indices[k] + 1]++;
    for (int c = 0; c < n; ++c) colPtr[c + 1] += colPtr[c];
    colRow.resize(A.col_indices.size());
    colK.resize(A.col_indices.size());
    {
        std::vector<int> fill(colPtr.begin(), colPtr.end() - 1);
        for (int i = 0; i < n; ++i)
            for (int k = A.row_pointer[i]; k < A.row_pointer[i + 1]; ++k) {
                int c = A.col_indices[k];
                colRow[fill[c]] = i;
                colK[fill[c]++] = k;
            }
    }
    std::vector<int> pos(n, -1);
    for (auto& f : plan.fronts) {
        const int m = f.dim();
        const int p = (int)f.pivots.size();
        for (int i = 0; i < p; ++i) pos[f.pivots[i]] = i;
        for (int i = 0; i < (int)f.boundary.size(); ++i) pos[f.boundary[i]] = p + i;
        // Working on B = P*A: B's row v is A's row rowOfVar[v]
        for (int v : f.pivots) {
            const int ar = plan.rowOfVar[v];
            for (int k = A.row_pointer[ar]; k < A.row_pointer[ar + 1]; ++k) {
                int c = A.col_indices[k];
                if (rank[c] < rank[v]) continue;
                f.assembleK.push_back(k);
                f.assembleRow.push_back(pos[v]);
                f.assembleCol.push_back(pos[c]);
            }
            for (int t = colPtr[v]; t < colPtr[v + 1]; ++t) {
                int rr = varOfRow[colRow[t]];
                if (rank[rr] <= rank[v]) continue;
                f.assembleK.push_back(colK[t]);
                f.assembleRow.push_back(pos[rr]);
                f.assembleCol.push_back(pos[v]);
            }
        }
        for (int c : f.children) {
            std::vector<int> map;
            for (int u : plan.fronts[c].boundary) map.push_back(pos[u]);
            f.childMap.push_back(std::move(map));
        }
        for (int v : f.pivots) pos[v] = -1;
        for (int v : f.boundary) pos[v] = -1;

        double dm = m;
        for (int k = 0; k < p; ++k) plan.flopEstimate += 2.0 * (dm - k - 1) * (dm - k - 1);
        plan.peakEntries = std::max(plan.peakEntries, dm * dm);
    }

    // Levels by height: fronts within a level share no data and run in parallel
    std::vector<int> height(plan.fronts.size(), 0);
    for (int i = 0; i < (int)plan.fronts.size(); ++i)
        for (int c : plan.fronts[i].children) height[i] = std::max(height[i], height[c] + 1);
    for (int i = 0; i < (int)plan.fronts.size(); ++i) {
        if ((int)plan.levels.size() <= height[i]) plan.levels.resize(height[i] + 1);
        plan.levels[height[i]].push_back(i);
    }
    return plan;
    };
    const char* forced = std::getenv("TREESPICE_ORDERING");
    const std::string only = forced ? forced : "";
    ContractionPlan best;
    if (only != "md") {
        Dissector nd{adj, std::max(1, leafSize), std::vector<int>(n, 0), 1, {}};
        const int root = hubs.empty() ? -1 : nd.addFront(hubs, -1);
        nd.dissect(rest, root);
        best = finish(std::move(nd.fronts));
    }
    if (only != "nd") {
        std::vector<ContractionFront> pre;
        if (!hubs.empty()) { pre.emplace_back(); pre.back().pivots = hubs; }
        const int offset = (int)pre.size();
        for (auto& f : minDegreeFronts(adj, rest)) {
            f.parent = f.parent >= 0 ? f.parent + offset : (hubs.empty() ? -1 : 0);
            pre.push_back(std::move(f));
        }
        ContractionPlan md = finish(std::move(pre));
        auto cost = [](const ContractionPlan& c) { return c.flopEstimate + 1e4 * (double)c.fronts.size(); };
        if (best.fronts.empty() || cost(md) < cost(best)) best = std::move(md);
    }
    return best;
}

namespace {

// Threshold for accepting a diagonal pivot inside a front: |a_kk| >= u * max
// |a_rk| over ALL rows of the front (MUMPS uses the same default, 0.01).
constexpr double kPivotThreshold = 0.01;
constexpr int kMaxDelayed = 8;  // rows a front may delay to its parent (see the factor loop)

// Per-front numeric data. Local layout: [own pivots p][delayed-in d][boundary b].
// After factorization the first `e` columns hold L (unit diagonal implied), the
// first `e` rows hold U, and positions e..m-1 (delayed-out + boundary) form the
// update tensor handed to the parent. Row and column labels can differ inside
// the delayed segment (unsymmetric pivoting); they always coincide on the
// boundary, and the counts match, so the parent pairs them by position.
struct FrontFactor {
    std::vector<double> M;
    std::vector<int> rowLab;  // equation (B row = variable whose equation it is) at each local row
    std::vector<int> colLab;  // variable at each local column
    std::vector<int> sw;      // row swapped into k at elimination step k
    std::vector<std::vector<int>> childPos;  // child trailing position -> local position
    int m = 0, e = 0, full = 0;  // dim, eliminated count, fully-summed count (p + d)
    bool singular = false;
    double minPivot = std::numeric_limits<double>::infinity();
    int minPivotRow = -1;  // equation (row of A) of the smallest pivot
};

// Work estimate for one level: factorization ~ sum dim^2 * pivots, solve ~ sum dim^2
double levelWork(const ContractionPlan& plan, const std::vector<int>& level, bool factor) {
    double w = 0.0;
    for (int id : level) {
        double m = plan.fronts[id].dim();
        w += factor ? m * m * (double)plan.fronts[id].pivots.size() : m * m;
    }
    return w;
}

// Dynamic layout of front `id` given its children's factors: fills lab,
// childPos and the sizes. Shared by factorization so solves reuse it.
void layoutFront(const ContractionPlan& plan, int id, std::vector<FrontFactor>& F) {
    const auto& fr = plan.fronts[id];
    auto& ff = F[id];
    const int p = (int)fr.pivots.size(), b = (int)fr.boundary.size();
    std::vector<int> delayedRows, delayedCols;
    for (int c : fr.children)
        for (int q = F[c].e; q < F[c].full; ++q) {
            delayedRows.push_back(F[c].rowLab[q]);
            delayedCols.push_back(F[c].colLab[q]);
        }
    const int d = (int)delayedRows.size();
    ff.full = p + d;
    ff.m = p + d + b;
    for (auto* lab : {&ff.rowLab, &ff.colLab}) {
        lab->assign(fr.pivots.begin(), fr.pivots.end());
        auto& del = (lab == &ff.rowLab) ? delayedRows : delayedCols;
        lab->insert(lab->end(), del.begin(), del.end());
        lab->insert(lab->end(), fr.boundary.begin(), fr.boundary.end());
    }
    ff.childPos.assign(fr.children.size(), {});
    int nextDelayed = p;
    for (size_t j = 0; j < fr.children.size(); ++j) {
        const auto& cf = F[fr.children[j]];
        auto& cp = ff.childPos[j];
        for (int q = cf.e; q < cf.full; ++q) cp.push_back(nextDelayed++);   // delayed-out of child
        for (int s : fr.childMap[j]) cp.push_back(s < p ? s : s + d);        // child boundary (static map)
    }
}

// Forward pass (rhs through the tree) + top-down back substitution.
void solveWithFactors(const ContractionPlan& plan, const std::vector<FrontFactor>& F,
                      const std::vector<double>& b, std::vector<double>& x) {
    const int nf = (int)plan.fronts.size();
    std::vector<std::vector<double>> w(nf);
    for (const auto& level : plan.levels) {
        parallelFor((int)level.size(), levelWork(plan, level, false), [&](int li) {
            const int id = level[li];
            const auto& fr = plan.fronts[id];
            const auto& ff = F[id];
            auto& v = w[id];
            v.assign(ff.m, 0.0);
            for (size_t i = 0; i < fr.pivots.size(); ++i) v[i] = b[plan.rowOfVar[fr.pivots[i]]];
            for (size_t j = 0; j < fr.children.size(); ++j) {  // superpose children's rhs updates
                const int c = fr.children[j];
                const auto& cp = ff.childPos[j];
                for (size_t a = 0; a < cp.size(); ++a) v[cp[a]] += w[c][F[c].e + a];
            }
            // Factorization swapped WHOLE rows (earlier L columns too, LAPACK
            // getrf style), so apply every swap first, then substitute with L.
            for (int k = 0; k < ff.e; ++k) std::swap(v[k], v[ff.sw[k]]);
            for (int k = 0; k < ff.e; ++k)
                for (int r = k + 1; r < ff.m; ++r) v[r] -= ff.M[(size_t)r * ff.m + k] * v[k];
        });
    }
    x.assign(plan.n, 0.0);
    for (int li = (int)plan.levels.size() - 1; li >= 0; --li) {
        const auto& level = plan.levels[li];
        parallelFor((int)level.size(), levelWork(plan, level, false), [&](int t) {
            const int id = level[t];
            const auto& ff = F[id];
            std::vector<double> xl(ff.m);
            for (int q = ff.e; q < ff.m; ++q) xl[q] = x[ff.colLab[q]];  // solved by ancestors
            for (int k = ff.e - 1; k >= 0; --k) {
                double s = w[id][k];
                for (int c = k + 1; c < ff.m; ++c) s -= ff.M[(size_t)k * ff.m + c] * xl[c];
                xl[k] = s / ff.M[(size_t)k * ff.m + k];
                x[ff.colLab[k]] = xl[k];
            }
        });
    }
}

}  // namespace

struct ContractionCache::Impl {
    const ContractionPlan* plan = nullptr;
    std::vector<FrontFactor> F;
    std::vector<std::vector<double>> in;  // per front: the A values it assembled last time
};

namespace {
// r = b - A x; returns the componentwise backward error max_i |r_i| / (|A||x| + |b|)_i
double residual(const Csr_matrix& A, const std::vector<double>& b, const std::vector<double>& x, std::vector<double>& r) {
    const int n = A.rows;
    r.resize(n);
    matrixMultiplication(A, x.data(), r.data());
    double omega = 0.0;
    for (int i = 0; i < n; ++i) {
        double scale = std::abs(b[i]);
        for (int k = A.row_pointer[i]; k < A.row_pointer[i + 1]; ++k)
            scale += std::abs(A.values[k] * x[A.col_indices[k]]);
        r[i] = b[i] - r[i];
        if (scale > 0.0) omega = std::max(omega, std::abs(r[i]) / scale);
        else if (r[i] != 0.0) omega = std::numeric_limits<double>::infinity();
    }
    return omega;
}

}  // namespace

namespace {
SolverResult solve(const Csr_matrix& A, const std::vector<double>& b, const ContractionPlan& plan,
                   int refineSteps, ContractionCache* cache, bool gpu) {
    static const int gpuMin = [] { const char* s = std::getenv("TREESPICE_GPU_FRONT"); return s ? std::atoi(s) : 512; }();
    SolverResult res;
    res.iterations = 1;
    const int n = A.rows;
    if (n == 0) { res.converged = true; res.finalResidual = 0.0; return res; }
    if (!plan.matches(A) || (int)b.size() != n) { res.converged = false; res.finalResidual = 1.0; return res; }

    // Factorization, bottom-up by level. Each front assembles its Jacobian
    // entries, adds its children's update tensors (superposition of Schur
    // complements) and eliminates what it safely can:
    //  - threshold partial pivoting: a pivot (fully-summed row i, column j) is
    //    accepted if |a_ij| >= u * max over ALL rows of the front of |a_rj|
    //    (boundary rows are not fully summed, so they can only veto, not pivot);
    //  - if no fully-summed pair passes, the remaining rows/columns are DELAYED
    //    to the parent, where more of their equations are summed;
    //  - fronts without a boundary (roots) have only fully-summed rows, so every
    //    column finds a pivot there.
    const int nf = (int)plan.fronts.size();
    std::vector<FrontFactor> local;
    bool warm = false;  // the cache holds factors of this plan
    if (cache) {
        if (!cache->impl) cache->impl = std::make_shared<ContractionCache::Impl>();
        auto& c = *cache->impl;
        warm = c.plan == &plan && (int)c.F.size() == nf;
        if (!warm) { c.plan = &plan; c.F.assign(nf, {}); c.in.assign(nf, {}); }
    } else local.resize(nf);
    std::vector<FrontFactor>& F = cache ? cache->impl->F : local;
    std::vector<char> reused(nf, 0), onGpu(nf, 0);
    bool singular = false;
    double smallestPivot = std::numeric_limits<double>::infinity();
    int smallestPivotRow = -1;
    for (const auto& level : plan.levels) {
        parallelFor((int)level.size(), levelWork(plan, level, true), [&](int li) {
            const int id = level[li];
            const auto& fr = plan.fronts[id];
            if (cache) {  // dirty flag: inputs and every child unchanged -> keep the factor
                auto& in = cache->impl->in[id];
                bool same = warm && in.size() == fr.assembleK.size();
                for (int c : fr.children) same = same && reused[c];
                for (size_t t = 0; same && t < in.size(); ++t)
                    same = std::memcmp(&in[t], &A.values[fr.assembleK[t]], sizeof(double)) == 0;
                if (same) { reused[id] = 1; return; }
                in.resize(fr.assembleK.size());
                for (size_t t = 0; t < in.size(); ++t) in[t] = A.values[fr.assembleK[t]];
            }
            layoutFront(plan, id, F);
            auto& ff = F[id];
            ff.e = 0;  // a cached factor object carries the last solve's state
            ff.singular = false;
            ff.minPivot = std::numeric_limits<double>::infinity();
            ff.minPivotRow = -1;
            const int m = ff.m, p = (int)fr.pivots.size(), d = ff.full - p;
            auto& M = ff.M;
            auto notePivot = [&](int k) {
                if (const double a = std::abs(M[(size_t)k * m + k]); a < ff.minPivot) {
                    ff.minPivot = a;
                    ff.minPivotRow = plan.rowOfVar[ff.rowLab[k]];
                }
            };
            // A cached front keeps its buffer between solves; one that grew in an earlier,
            // near-singular iterate (delayed rows) gives the excess back, or every front keeps
            // its worst size for good (14 GB at 4e5 fronts of a 1e5-stage chain)
            if (M.capacity() > 4 * (size_t)m * m + 64) std::vector<double>().swap(M);
            M.assign((size_t)m * m, 0.0);
            for (size_t t = 0; t < fr.assembleK.size(); ++t) {
                int r = fr.assembleRow[t], c = fr.assembleCol[t];
                if (r >= p) r += d;
                if (c >= p) c += d;
                M[(size_t)r * m + c] += A.values[fr.assembleK[t]];
            }
            for (size_t j = 0; j < fr.children.size(); ++j) {
                const auto& cf = F[fr.children[j]];
                const auto& cp = ff.childPos[j];
                for (size_t a = 0; a < cp.size(); ++a)
                    for (size_t bb = 0; bb < cp.size(); ++bb)
                        M[(size_t)cp[a] * m + cp[bb]] += cf.M[(size_t)(cf.e + a) * cf.m + (cf.e + bb)];
            }

            // Large fronts: the GPU eliminates in FP32 until a pivot fails the threshold;
            // the CPU continues from there (column search, delayed pivots) in FP64.
            ff.sw.resize(ff.full);
            const int k0 = gpu && gpuMin > 0 && m >= gpuMin ? gpuFrontLu(M.data(), m, ff.full, kPivotThreshold, ff.sw.data()) : 0;
            for (int k = 0; k < k0; ++k) {
                std::swap(ff.rowLab[k], ff.rowLab[ff.sw[k]]);
                notePivot(k);
            }
            ff.sw.resize(k0);
            ff.e = k0;
            onGpu[id] = k0 > 0;
            for (int k = k0; k < ff.full; ++k) {
                // Column search: first column j (from k) whose best fully-summed
                // row passes the threshold against the whole column.
                int pr = -1, pc = -1;
                double bestRatio = 0.0;
                for (int j = k; j < ff.full && pc < 0; ++j) {
                    double colmax = 0.0, rowBest = 0.0;
                    int ri = -1;
                    for (int r = k; r < m; ++r) {
                        double a = std::abs(M[(size_t)r * m + j]);
                        colmax = std::max(colmax, a);
                        if (r < ff.full && a > rowBest) { rowBest = a; ri = r; }
                    }
                    double ratio = colmax > 0.0 ? rowBest / colmax : 0.0;
                    if (ri >= 0 && ratio >= kPivotThreshold) { pr = ri; pc = j; }
                    bestRatio = std::max(bestRatio, ratio);
                }
                // Delayed rows must not snowball up the tree: a near-singular Newton
                // iterate of a 30k-stage BSIM4 chain delayed 2e5 pivots into fronts 673
                // wide (near O(n^2) memory, 131 GB at 1e5 stages). Past kMaxDelayed rows,
                // the largest fully-summed entry pivots without the boundary rows' veto;
                // if they are all zero the matrix is singular here: say so now instead of
                // carrying the zero rows up the tree (they reached fronts ~100 wide).
                if (pc < 0 && ff.full - k > kMaxDelayed) {
                    double best = 0.0;
                    for (int j = k; j < ff.full; ++j)
                        for (int r = k; r < ff.full; ++r)
                            if (const double a = std::abs(M[(size_t)r * m + j]); a > best) { best = a; pr = r; pc = j; }
                    if (pc < 0) { ff.singular = true; break; }
                }
                if (pc < 0) break;  // delay the remaining rows/columns to the parent
                if (pc != k) {
                    for (int r = 0; r < m; ++r) std::swap(M[(size_t)r * m + k], M[(size_t)r * m + pc]);
                    std::swap(ff.colLab[k], ff.colLab[pc]);
                }
                if (pr != k) {
                    for (int c = 0; c < m; ++c) std::swap(M[(size_t)k * m + c], M[(size_t)pr * m + c]);
                    std::swap(ff.rowLab[k], ff.rowLab[pr]);
                }
                ff.sw.push_back(pr);
                const double piv = M[(size_t)k * m + k];
                notePivot(k);
                if (!(std::abs(piv) > 0.0) || !std::isfinite(piv)) { ff.singular = true; break; }
                const double inv = 1.0 / piv;
                for (int r = k + 1; r < m; ++r) {
                    double& l = M[(size_t)r * m + k];
                    if (l == 0.0) continue;
                    l *= inv;
                    for (int c = k + 1; c < m; ++c) M[(size_t)r * m + c] -= l * M[(size_t)k * m + c];
                }
                ff.e = k + 1;
            }
        });
        for (int id : level) {  // reduce per-front results after the level
            // A root front has no parent to delay to: un-eliminated fully-summed rows
            // mean no pivot exists (e.g. a node with no DC path). They used to be left
            // silently at x = 0 and the solve reported success.
            if (plan.fronts[id].parent < 0 && F[id].e < F[id].full) F[id].singular = true;
            if (F[id].singular && res.rankDeficientRow < 0) res.rankDeficientRow = plan.rowOfVar[F[id].rowLab[F[id].e]];
            singular = singular || F[id].singular;
            if (F[id].minPivot < smallestPivot) {
                smallestPivot = F[id].minPivot;
                smallestPivotRow = F[id].minPivotRow;
            }
        }
        if (singular) break;
    }
    res.smallestPivot = smallestPivot;
    if (res.rankDeficientRow < 0) res.rankDeficientRow = smallestPivotRow;  // linalg.h: row of the smallest pivot
    if (cache) {
        for (char r : reused) (r ? cache->reused : cache->refactored)++;
        if (singular) cache->impl.reset();  // never reuse a failed factorization
    }
    if (singular) {
        res.converged = false;
        res.rankDeficient = true;
        res.finalResidual = 1.0;
        res.solution.assign(n, 0.0);
        return res;
    }

    std::vector<double> x, r(n), dx;
    solveWithFactors(plan, F, b, x);

    // FP32 fronts need refinement whatever the caller asked; it converges linearly
    // (rate ~ kappa * 2^-24), so stop once a step no longer halves the error.
    constexpr double eps = std::numeric_limits<double>::epsilon();
    res.gpuFronts = (int)std::count(onGpu.begin(), onGpu.end(), 1);
    const int steps = res.gpuFronts ? std::max(refineSteps, 10) : refineSteps;
    double omega = residual(A, b, x, r);
    for (int s = 0; s < steps && omega > 4.0 * eps; ++s) {
        solveWithFactors(plan, F, r, dx);
        for (int i = 0; i < n; ++i) x[i] += dx[i];
        const double prev = omega;
        omega = residual(A, b, x, r);
        res.iterations++;
        if (res.gpuFronts && !(omega < 0.5 * prev)) break;
    }
    if (res.gpuFronts && !(omega <= 64.0 * eps)) {  // FP32 factors too weak here: redo in FP64
        if (cache) cache->impl.reset();
        res = solve(A, b, plan, refineSteps, cache, false);
        res.fp64Refactor = true;
        return res;
    }

    double r2 = 0.0;
    for (double v : r) r2 += v * v;
    res.solution = std::move(x);
    res.finalResidual = std::sqrt(r2);
    res.converged = std::isfinite(res.finalResidual);
    return res;
}
}  // namespace

SolverResult solveContraction(const Csr_matrix& A, const std::vector<double>& b,
                              const ContractionPlan& plan, int refineSteps, ContractionCache* cache, bool gpu) {
    return solve(A, b, plan, refineSteps, cache, gpu);
}
