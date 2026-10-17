#pragma once
// Persistent worker pool and parallelFor, shared by the contraction solver (fronts of
// one tree level) and device evaluation (OSDI instances of one Newton iteration).
// Without ACUTESIM_THREADS_ENABLED (set for the engine's own objects) everything runs
// serially in the caller.
#include <functional>

#ifdef ACUTESIM_THREADS_ENABLED
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

// Persistent workers for parallelFor, created on first use (spawning threads per
// level was ~15 % of an hcircuit solve). One level runs at a time; a caller that
// finds the pool busy (another solve on another thread, e.g. parallel Monte Carlo)
// runs its level serially instead of waiting. TREESPICE_THREADS caps the size.
class WorkerPool {
public:
    static WorkerPool& get() { static WorkerPool p; return p; }
    bool parallel() const { return !threads_.empty(); }
    bool tryRun(int n, const std::function<void(int)>& fn) {
        std::unique_lock<std::mutex> busy(job_, std::try_to_lock);
        if (!busy) return false;
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &fn; n_ = n; next_ = 0; active_ = (int)threads_.size(); ++gen_;
        }
        wake_.notify_all();
        work();
        std::unique_lock<std::mutex> lk(m_);
        done_.wait(lk, [&] { return active_ == 0; });
        return true;
    }

private:
    WorkerPool() {
        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
#ifdef __APPLE__
        // Apple Silicon: efficiency cores stall the level barrier; use the performance cores only
        int perf = 0; size_t len = sizeof perf;
        if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, nullptr, 0) == 0 && perf > 0) hw = (unsigned)perf;
#endif
        if (const char* t = std::getenv("TREESPICE_THREADS"))  // e.g. 1 to compare with serial solvers
            hw = (unsigned)std::max(1, std::atoi(t));
        for (unsigned t = 1; t < hw; ++t) threads_.emplace_back([this] { loop(); });
    }
    ~WorkerPool() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; ++gen_; }
        wake_.notify_all();
        for (auto& t : threads_) t.join();
    }
    void work() { for (int i; (i = next_.fetch_add(1)) < n_;) (*fn_)(i); }
    void loop() {
        unsigned long seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                wake_.wait(lk, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
            }
            work();
            std::lock_guard<std::mutex> lk(m_);
            if (--active_ == 0) done_.notify_one();
        }
    }
    std::vector<std::thread> threads_;
    std::mutex job_, m_;
    std::condition_variable wake_, done_;
    const std::function<void(int)>* fn_ = nullptr;
    int n_ = 0, active_ = 0;
    std::atomic<int> next_{0};
    unsigned long gen_ = 0;
    bool stop_ = false;
};
#endif

// Run fn(i) for i in [0, n) on the worker pool when the level carries enough work
// (a few us to hand out); serial otherwise. Fronts in a level share no writes and
// the result is independent of scheduling.
template <class Fn>
inline void parallelFor(int n, double work, Fn&& fn) {
#ifdef ACUTESIM_THREADS_ENABLED
    constexpr double kMinParallelWork = 2e6;  // flops-ish
    if (n > 1 && work >= kMinParallelWork && WorkerPool::get().parallel() &&
        WorkerPool::get().tryRun(n, std::function<void(int)>(std::ref(fn))))
        return;
#endif
    for (int i = 0; i < n; ++i) fn(i);
}

