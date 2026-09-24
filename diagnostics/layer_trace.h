#pragma once

// 诊断插桩: 每层 cudaEvent 记录(cuda graph 捕获 event 节点, replay 时更新时间戳)
// 由环境变量 NINFER_LAYER_TRACE 启用, 每 stats 周期 dump 各层耗时分布

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>

namespace ninfer::diag::layer_trace {

inline bool enabled() {
    static const bool value = std::getenv("NINFER_LAYER_TRACE") != nullptr;
    return value;
}

inline cudaEvent_t event(std::size_t index) {
    static std::array<cudaEvent_t, 130> events = [] {
        std::array<cudaEvent_t, 130> e{};
        for (auto& ev : e) { cudaEventCreate(&ev); }
        return e;
    }();
    return events[index];
}

inline void dump(int n_layers) {
    std::fprintf(stderr, "layer_trace |");
    float total = 0.0f;
    for (int l = 0; l < n_layers; ++l) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, event(static_cast<std::size_t>(l)),
                             event(static_cast<std::size_t>(l) + 1));
        total += ms;
        std::fprintf(stderr, " %d:%.3f", l, ms);
    }
    float replay_ms = -1.0f;
    cudaEventElapsedTime(&replay_ms, event(128), event(129));
    std::fprintf(stderr, " | total %.3f ms | graph_replay %.3f ms\n", total, replay_ms);
}

} // namespace ninfer::diag::layer_trace
