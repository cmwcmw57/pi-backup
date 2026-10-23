// ================================================================
// map_export.h
// 설명: reset으로 여러 과거 ID에 남은 동일 Map 객체를 한 번만 export하도록 선택한다.
// 사용법: rover::unique_map_pointers(observed_maps)
// ================================================================
#pragma once
#include <set>
#include <vector>

namespace rover {
template<typename MapIndex>
std::vector<typename MapIndex::mapped_type> unique_map_pointers(const MapIndex& maps) {
    std::vector<typename MapIndex::mapped_type> result;
    std::set<typename MapIndex::mapped_type> seen;
    for (const auto& item : maps)
        if (seen.insert(item.second).second) result.push_back(item.second);
    return result;
}
}
