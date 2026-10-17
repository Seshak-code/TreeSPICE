  std::string auditPhysicalResiduals(const TensorNetlist &netlist, const std::vector<double> &v_candidate) {
    std::string report;
    auto getV = [&](int nodeIdx) {
      if (nodeIdx <= 0 || nodeIdx > (int)v_candidate.size()) return 0.0;
      return v_candidate[nodeIdx - 1];
    };

    int n = netlist.numGlobalNodes;
    std::vector<double> node_residuals(n, 0.0);
    auto addRes = [&](int nodeIdx, double val) {
      if (nodeIdx > 0 && nodeIdx <= n) node_residuals[nodeIdx - 1] += val;
    };

    // AUDIT 1: Branch elements (V, L, E, F, H): node-row terms of the shared
    // branch system; V-source currents are solved unknowns in v_candidate
    {
      BranchSystem S;
      buildBranchSystem(netlist, 0.0, 0.0, S);
      for (size_t t = 0; t < S.r.size(); ++t)
        if (S.r[t] < n && S.c[t] < (int)v_candidate.size())
          node_residuals[S.r[t]] -= S.v[t] * v_candidate[S.c[t]];
      for (size_t i = 0; i < netlist.globalBlock.voltageSources.size(); ++i) {
        const int b = n + (int)i;  // global V sources are the first branch rows
        const double i_src = b < (int)v_candidate.size() ? v_candidate[b] : 0.0;
        if (std::abs(i_src) > 100.0) {
          const auto &v = netlist.globalBlock.voltageSources[i];
          report += "WARN: VSource (Nodes " + std::to_string(v.nodePositive) + "," + std::to_string(v.nodeNegative) + ") I=" + std::to_string(i_src) + " A\n";
        }
      }
    }

    // AUDIT 2: Current Sources
    for (const auto &cs : netlist.globalBlock.currentSources) {
      addRes(cs.nodePositive, -cs.current_A);
      addRes(cs.nodeNegative, cs.current_A);
    }

    // AUDIT 3: Resistors
    for (const auto &r : netlist.globalBlock.resistors) {
       double v1 = getV(r.nodeTerminal1);
       double v2 = getV(r.nodeTerminal2);
       double i_r = (v1 - v2) / r.resistance_ohms;
       addRes(r.nodeTerminal1, -i_r);
       addRes(r.nodeTerminal2, i_r);
    }

    // AUDIT 4: Diodes and BJTs (DC junctionCurrents(); terminal currents > 10 A are reported)
    forEachJunction(netlist, [&](const Junction &j) {
        std::array<double, 7> x{};
        for (int k = 0; k < j.terms; ++k) x[k] = getV(j.n[k]);
        const auto I = junctionCurrents(j, x, 0.0);
        const std::string &name = j.d ? j.d->instanceName : j.f ? j.f->instanceName : j.q->instanceName;
        for (int k = 0; k < j.terms; ++k) {
            addRes(j.n[k], -I[k]);
            if (std::abs(I[k]) > 10.0)
                report += "WARN: " + name + " terminal " + std::to_string(k) + " I=" + std::to_string(I[k]) + " A\n";
        }
    });

    // AUDIT 6: KCL Check
    for (int i=0; i<n; ++i) {
        if (std::abs(node_residuals[i]) > 10.0) {
            report += "FAIL: Node " + std::to_string(i+1) + " KCL Resid=" + std::to_string(node_residuals[i]) + " A. V=" + std::to_string(getV(i+1)) + "\n";
        }
    }

    // Capture Partial Voltages
    if (report.length() > 0) {
        report += "Voltages: ";
        for(size_t i=0; i<v_candidate.size(); ++i) {
            if (i >= 10) { report += "..."; break; } 
            report += std::to_string(v_candidate[i]) + ", ";
        }
        report += "\n";
    }

    return report;
  }
