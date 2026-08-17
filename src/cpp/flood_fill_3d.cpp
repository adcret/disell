// flood_fill_3d.cpp
//
// 3D (and 2D-lifted) region-growing segmentation for DFXM orientation fields.
//
//
// Conventions:
//   property_map : (Z, Y, X, C) float32, channel-last.
//   footprint    : (FZ, FY, FX) bool.
//   mask         : (Z, Y, X) uint8/bool. REQUIRED. Marks the valid domain.
//                  The caller MUST exclude NaN voxels from the mask; this code
//                  does not test for NaN.
//   thresholds   : same physical unit as the property field. The criterion is
//                  sum-of-squares: dist^2 < threshold^2 * C  (per-channel).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// ------------------------------------------------------------------ helpers

struct Offset3D {
    int dz, dy, dx;
    Offset3D(int dz_, int dy_, int dx_) : dz(dz_), dy(dy_), dx(dx_) {}
};

static inline size_t linear_index(int z, int y, int x, int Y, int X) {
    return (size_t)z * (size_t)Y * (size_t)X + (size_t)y * (size_t)X + (size_t)x;
}

static inline bool in_bounds(int z, int y, int x, int Z, int Y, int X) {
    return (z >= 0 && z < Z && y >= 0 && y < Y && x >= 0 && x < X);
}

static std::vector<Offset3D> build_offsets(const bool* fdata,
                                           int FZ, int FY, int FX) {
    const int mZ = FZ / 2, mY = FY / 2, mX = FX / 2;
    std::vector<Offset3D> offsets;
    offsets.reserve((size_t)FZ * FY * FX);
    for (int dz = 0; dz < FZ; ++dz)
        for (int dy = 0; dy < FY; ++dy)
            for (int dx = 0; dx < FX; ++dx)
                if (fdata[((size_t)dz * FY + dy) * FX + dx])
                    offsets.emplace_back(dz - mZ, dy - mY, dx - mX);
    if (offsets.empty())
        throw std::runtime_error("footprint contains no active voxels");
    return offsets;
}

// ------------------------------------------------------- single region grow
//
// Grows one connected region from one seed using a DFS over the footprint
// neighbourhood. A voxel is accepted into the region if it is the seed, OR if
// at least ceil(footprint_tolerance * valid_neighbours) of its in-mask
// neighbours pass the local (and, if enabled, global running-mean) test.
//
// `visited` is supplied by the caller and is assumed to be all-zero on entry;
// it is left all-zero on return (every set entry is reset before returning),
// so the caller can reuse one buffer across many seeds without reallocating
// Nvox bytes each time.
void flood_fill_single_region_binary_3d(
    const float* __restrict prop,
    const uint8_t* __restrict mask,
    int Z, int Y, int X, int C,
    const std::vector<Offset3D>& offsets,
    int si, int sj, int sk,
    float thr_sq_C,
    float global_threshold,
    float footprint_tolerance,
    std::vector<uint8_t>& visited,        // scratch, size Nvox, all-zero in/out
    std::vector<size_t>& region_indices   // output
) {
    region_indices.clear();

    if (!in_bounds(si, sj, sk, Z, Y, X)) return;

    const size_t seed_idx = linear_index(si, sj, sk, Y, X);
    if (!mask[seed_idx]) return;

    const float* seed_feat = prop + seed_idx * (size_t)C;
    const bool global_enabled = global_threshold > 0.0f;
    const float g_thr_sq_C = global_threshold * global_threshold * float(C);

    std::vector<float> region_mean;
    if (global_enabled) {
        region_mean.assign(C, 0.0f);
        for (int c = 0; c < C; ++c) region_mean[c] = seed_feat[c];
    }

    const size_t stride_yx = (size_t)Y * X;

    std::vector<size_t> stack;
    stack.reserve(8192);
    stack.push_back(seed_idx);
    visited[seed_idx] = 1;

    // Track everything we marked visited so we can zero it again at the end.
    std::vector<size_t> touched;
    touched.reserve(8192);
    touched.push_back(seed_idx);

    std::vector<size_t> candidates;
    candidates.reserve(offsets.size());

    while (!stack.empty()) {
        const size_t idx = stack.back();
        stack.pop_back();

        const int z = (int)(idx / stride_yx);
        const int y = (int)((idx % stride_yx) / (size_t)X);
        const int x = (int)(idx % (size_t)X);

        const float* center = prop + idx * (size_t)C;

        candidates.clear();
        int valid_neighbors = 0;
        int count_pass = 0;

        for (const auto& off : offsets) {
            const int nz = z + off.dz, ny = y + off.dy, nx = x + off.dx;
            if (!in_bounds(nz, ny, nx, Z, Y, X)) continue;     // safety guard

            const size_t nidx = linear_index(nz, ny, nx, Y, X);
            if (!mask[nidx]) continue;
            valid_neighbors++;

            const float* neigh = prop + nidx * (size_t)C;
            float dist2 = 0.0f;
            #pragma omp simd reduction(+:dist2)
            for (int c = 0; c < C; ++c) {
                const float d = neigh[c] - center[c];
                dist2 += d * d;
            }
            bool pass = dist2 < thr_sq_C;

            if (pass && global_enabled) {
                float ds2 = 0.0f;
                #pragma omp simd reduction(+:ds2)
                for (int c = 0; c < C; ++c) {
                    const float ds = neigh[c] - region_mean[c];
                    ds2 += ds * ds;
                }
                pass = ds2 < g_thr_sq_C;
            }
            if (pass) { count_pass++; candidates.push_back(nidx); }
        }

        const int min_pass =
            (int)std::ceil(footprint_tolerance * (float)valid_neighbors);
        const bool is_seed = (idx == seed_idx);

        if (is_seed || count_pass >= min_pass) {
            region_indices.push_back(idx);

            if (global_enabled) {
                const size_t n = region_indices.size();
                for (int c = 0; c < C; ++c)
                    region_mean[c] =
                        (region_mean[c] * float(n - 1) + center[c]) / float(n);
            }
            for (size_t nidx : candidates) {
                if (!visited[nidx]) {
                    visited[nidx] = 1;
                    touched.push_back(nidx);
                    stack.push_back(nidx);
                }
            }
        }
    }

    // Reset the scratch buffer to all-zero for the next seed.
    for (size_t t : touched) visited[t] = 0;
}

// ---------------------------------------------------------- main driver

py::dict flood_fill_random_seeds_3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> property_map,
    py::array_t<bool,  py::array::c_style | py::array::forcecast> footprint,
    float local_threshold,
    float global_threshold,
    float footprint_tolerance,
    py::object mask_obj,
    int max_iterations,
    int min_grain_size,
    bool fill_remaining,            // was `recycle_small_grains` (dead); now meaningful
    int stagnation_tolerance,
    py::object seed_points_obj = py::none(),
    int random_seed = -1
) {
    auto pbuf = property_map.request();
    if (pbuf.ndim != 4)
        throw std::runtime_error("property_map must be 4D (Z,Y,X,C)");
    const int Z = (int)pbuf.shape[0], Y = (int)pbuf.shape[1];
    const int X = (int)pbuf.shape[2], C = (int)pbuf.shape[3];
    const size_t Nvox = (size_t)Z * Y * X;
    const float* prop = static_cast<const float*>(pbuf.ptr);

    auto fbuf = footprint.request();
    if (fbuf.ndim != 3)
        throw std::runtime_error("footprint must be 3D (FZ,FY,FX)");
    const bool* fdata = static_cast<const bool*>(fbuf.ptr);
    const std::vector<Offset3D> offsets =
        build_offsets(fdata, (int)fbuf.shape[0], (int)fbuf.shape[1], (int)fbuf.shape[2]);

    if (mask_obj.is_none())
        throw std::runtime_error("mask must be provided (Python must compute it)");
    auto mask_arr =
        mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
    auto mbuf = mask_arr.request();
    if (mbuf.ndim != 3 || mbuf.shape[0] != Z || mbuf.shape[1] != Y || mbuf.shape[2] != X)
        throw std::runtime_error("mask must be shape (Z,Y,X) uint8/bool");
    uint8_t* mask = static_cast<uint8_t*>(mbuf.ptr);

    py::array_t<int> seg_arr({Z, Y, X});
    int* segmentation = seg_arr.mutable_data();
    std::fill_n(segmentation, Nvox, 0);

    std::vector<size_t> label_sizes;
    std::vector<std::vector<double>> label_means;

    const float thr_sq_C = local_threshold * local_threshold * float(C);

    std::mt19937 rng(random_seed >= 0 ? (unsigned)random_seed
                                       : std::random_device{}());

    // Remaining-pool with O(1) removal.
    std::vector<size_t> remaining;
    remaining.reserve(Nvox);
    std::vector<size_t> position_in_remaining(Nvox, (size_t)-1);
    for (size_t idx = 0; idx < Nvox; ++idx)
        if (mask[idx]) {
            position_in_remaining[idx] = remaining.size();
            remaining.push_back(idx);
        }

    auto remove_voxel = [&](size_t idx) {
        size_t pos = position_in_remaining[idx];
        if (pos == (size_t)-1) return;
        size_t last = remaining.back();
        remaining[pos] = last;
        position_in_remaining[last] = pos;
        remaining.pop_back();
        position_in_remaining[idx] = (size_t)-1;
    };

    // Parse user seeds (consumed LIFO so a size-ascending Python sort means
    // largest-first processing).
    std::vector<size_t> user_seeds;
    if (!seed_points_obj.is_none()) {
        auto seeds_arr =
            seed_points_obj.cast<py::array_t<long long, py::array::c_style | py::array::forcecast>>();
        auto sbuf = seeds_arr.request();
        if (sbuf.ndim != 2 || sbuf.shape[1] != 3)
            throw std::runtime_error("seed_points must be array of shape (N, 3)");
        const long long* sptr = static_cast<const long long*>(sbuf.ptr);
        const size_t Nseeds = (size_t)sbuf.shape[0];
        user_seeds.reserve(Nseeds);
        for (size_t i = 0; i < Nseeds; ++i) {
            const long long z = sptr[i*3+0], y = sptr[i*3+1], x = sptr[i*3+2];
            if (z < 0 || y < 0 || x < 0 || z >= Z || y >= Y || x >= X) {
                py::print("user seed out of bounds:", z, y, x);
                continue;
            }
            const size_t idx = linear_index((int)z, (int)y, (int)x, Y, X);
            if (mask[idx]) user_seeds.push_back(idx);
        }
    }
    const bool had_user_seeds = !user_seeds.empty();

    std::vector<uint8_t> visited(Nvox, 0);     // reusable scratch
    std::vector<size_t> region_indices;
    region_indices.reserve(8192);

    int label = 1;
    int iteration = 0;
    int last_success = -1;
    int user_seeds_processed = 0;
    int user_seeds_skipped_claimed = 0;
    int small_regions_parked = 0;

    while (iteration < max_iterations) {
        if (remaining.empty()) break;

        // --- pick a seed -------------------------------------------------
        size_t seed_idx = (size_t)-1;
        while (!user_seeds.empty()) {
            const size_t cand = user_seeds.back();
            user_seeds.pop_back();
            if (cand < Nvox && mask[cand] &&
                position_in_remaining[cand] != (size_t)-1) {
                seed_idx = cand;
                user_seeds_processed++;
                break;
            }
            user_seeds_skipped_claimed++;
        }
        if (seed_idx == (size_t)-1) {
            // No (more) usable user seeds. Only continue with random seeds if
            // either we were in pure-random mode (no user seeds were ever
            // given) OR the caller explicitly asked to fill the rest.
            if (had_user_seeds && !fill_remaining) break;
            std::uniform_int_distribution<size_t> dist(0, remaining.size() - 1);
            seed_idx = remaining[dist(rng)];
        }

        const int z = (int)(seed_idx / ((size_t)Y * X));
        const int y = (int)((seed_idx / (size_t)X) % (size_t)Y);
        const int x = (int)(seed_idx % (size_t)X);

        flood_fill_single_region_binary_3d(
            prop, mask, Z, Y, X, C, offsets, z, y, x,
            thr_sq_C, global_threshold, footprint_tolerance,
            visited, region_indices);

        const size_t grain_size = region_indices.size();

        // Seed is always accepted, so grain_size >= 1 for an in-mask seed.
        // grain_size == 0 only if the seed was already claimed/removed.
        if (grain_size == 0) {
            remove_voxel(seed_idx);     // never resample a dead seed
            iteration++;
            continue;
        }

        // --- small region: PARK it (do not delete, do not relabel) -------
        // Setting min_grain_size <= 0 disables this entirely (max coverage).
        if (min_grain_size > 0 && grain_size < (size_t)min_grain_size) {
            // Remove from the random pool so we don't resample the same tiny
            // region forever, but DO NOT clear the mask: leave these voxels
            // unlabelled (0) so a downstream watershed can still claim them.
            for (size_t idx : region_indices) remove_voxel(idx);
            small_regions_parked++;
            iteration++;
            continue;
        }

        // --- accept as a new label ---------------------------------------
        std::vector<double> mean_feat(C, 0.0);
        for (size_t idx : region_indices) {
            const float* f = prop + idx * (size_t)C;
            for (int c = 0; c < C; ++c) mean_feat[c] += f[c];
        }
        for (int c = 0; c < C; ++c) mean_feat[c] /= double(grain_size);

        const int new_label = label++;
        for (size_t idx : region_indices) {
            segmentation[idx] = new_label;
            mask[idx] = 0;            // claimed: remove from valid domain
            remove_voxel(idx);
        }
        label_sizes.push_back(grain_size);
        label_means.push_back(mean_feat);
        last_success = iteration;
        iteration++;

        if (stagnation_tolerance > 0 &&
            (iteration - last_success) > stagnation_tolerance)
            break;
    }

    // --- relabel to a contiguous 1..K (all stored labels already pass) ---
    const int num_labels_raw = label - 1;
    std::vector<int> new_map((size_t)std::max(num_labels_raw, 0), 0);
    int new_id = 1;
    for (int lbl = 1; lbl <= num_labels_raw; ++lbl)
        new_map[lbl - 1] = new_id++;
    const int num_final = new_id - 1;

    for (size_t idx = 0; idx < Nvox; ++idx) {
        const int old = segmentation[idx];
        if (old > 0) segmentation[idx] = new_map[old - 1];
    }

    py::array_t<double> means_arr({num_final, C});
    py::array_t<long long> sizes_arr({num_final});
    double* mp = static_cast<double*>(means_arr.mutable_data());
    long long* sp = static_cast<long long*>(sizes_arr.mutable_data());
    for (int lbl = 1; lbl <= num_labels_raw; ++lbl) {
        const int out = new_map[lbl - 1] - 1;
        if (out < 0) continue;
        sp[out] = (long long)label_sizes[lbl - 1];
        for (int c = 0; c < C; ++c)
            mp[(size_t)out * C + c] = label_means[lbl - 1][c];
    }

    py::dict out;
    out["segmentation"] = seg_arr;
    out["means"] = means_arr;
    out["sizes"] = sizes_arr;
    out["iterations"] = iteration;
    out["max_iterations_reached"] = (iteration >= max_iterations && !remaining.empty());
    out["remaining_voxels"] = (long long)remaining.size();
    out["user_seeds_supplied"] = (long long)(user_seeds_processed + user_seeds_skipped_claimed + user_seeds.size());
    out["user_seeds_processed"] = user_seeds_processed;
    out["user_seeds_skipped_claimed"] = user_seeds_skipped_claimed;
    out["user_seeds_unconsumed"] = (long long)user_seeds.size();
    out["small_regions_parked"] = small_regions_parked;
    return out;
}

// ---------------------------------------------------- seed collection (stage 1)

py::dict flood_fill_collect_seeds(
    py::array_t<float, py::array::c_style | py::array::forcecast> property_map,
    py::array_t<bool,  py::array::c_style | py::array::forcecast> footprint,
    float local_threshold,
    float global_threshold,
    float footprint_tolerance,
    py::object mask_obj,
    int max_iterations,
    int min_grain_size,
    int random_seed = -1
) {
    auto pbuf = property_map.request();
    if (pbuf.ndim != 4)
        throw std::runtime_error("property_map must be 4D (Z,Y,X,C)");
    const int Z = (int)pbuf.shape[0], Y = (int)pbuf.shape[1];
    const int X = (int)pbuf.shape[2], C = (int)pbuf.shape[3];
    const size_t Nvox = (size_t)Z * Y * X;
    const float* prop = static_cast<const float*>(pbuf.ptr);

    auto fbuf = footprint.request();
    if (fbuf.ndim != 3)
        throw std::runtime_error("footprint must be 3D (FZ,FY,FX)");
    const bool* fdata = static_cast<const bool*>(fbuf.ptr);
    const std::vector<Offset3D> offsets =
        build_offsets(fdata, (int)fbuf.shape[0], (int)fbuf.shape[1], (int)fbuf.shape[2]);

    if (mask_obj.is_none())
        throw std::runtime_error("mask must be provided");
    auto mask_arr =
        mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
    auto mbuf = mask_arr.request();
    if (mbuf.ndim != 3 || mbuf.shape[0] != Z || mbuf.shape[1] != Y || mbuf.shape[2] != X)
        throw std::runtime_error("mask must be shape (Z,Y,X)");
    const uint8_t* mask_in = static_cast<const uint8_t*>(mbuf.ptr);

    // Work on a private availability copy so the caller's mask is untouched
    // (the Python two-stage wrapper reuses it for the final fill).
    std::vector<uint8_t> available(mask_in, mask_in + Nvox);

    std::vector<size_t> remaining;
    remaining.reserve(Nvox);
    std::vector<size_t> position_in_remaining(Nvox, (size_t)-1);
    for (size_t idx = 0; idx < Nvox; ++idx)
        if (available[idx]) {
            position_in_remaining[idx] = remaining.size();
            remaining.push_back(idx);
        }

    auto remove_voxel = [&](size_t idx) {
        size_t pos = position_in_remaining[idx];
        if (pos == (size_t)-1) return;
        size_t last = remaining.back();
        remaining[pos] = last;
        position_in_remaining[last] = pos;
        remaining.pop_back();
        position_in_remaining[idx] = (size_t)-1;
        available[idx] = 0;
    };

    std::vector<long long> region_sizes, seed_points;
    region_sizes.reserve((size_t)std::max(max_iterations, 0));
    seed_points.reserve((size_t)std::max(max_iterations, 0) * 3);

    const float thr_sq_C = local_threshold * local_threshold * float(C);
    std::mt19937 rng(random_seed >= 0 ? (unsigned)random_seed
                                       : std::random_device{}());

    std::vector<uint8_t> visited(Nvox, 0);
    std::vector<size_t> region_indices;
    region_indices.reserve(8192);

    int iteration = 0;
    while (iteration < max_iterations) {
        if (remaining.empty()) break;

        std::uniform_int_distribution<size_t> dist(0, remaining.size() - 1);
        const size_t seed_idx = remaining[dist(rng)];
        const int z = (int)(seed_idx / ((size_t)Y * X));
        const int y = (int)((seed_idx / (size_t)X) % (size_t)Y);
        const int x = (int)(seed_idx % (size_t)X);

        flood_fill_single_region_binary_3d(
            prop, available.data(), Z, Y, X, C, offsets, z, y, x,
            thr_sq_C, global_threshold, footprint_tolerance,
            visited, region_indices);

        const size_t grain_size = region_indices.size();
        if (grain_size == 0) { remove_voxel(seed_idx); iteration++; continue; }

        if (min_grain_size <= 0 || grain_size >= (size_t)min_grain_size) {
            region_sizes.push_back((long long)grain_size);
            seed_points.push_back((long long)z);
            seed_points.push_back((long long)y);
            seed_points.push_back((long long)x);
        }
        // Consume the whole region so we don't resample it (or its members).
        for (size_t idx : region_indices) remove_voxel(idx);
        iteration++;
    }

    const size_t num = region_sizes.size();
    py::array_t<long long> sizes_arr((py::ssize_t)num);
    py::array_t<long long> seeds_arr(py::array::ShapeContainer{(py::ssize_t)num, (py::ssize_t)3});
    if (num > 0) {
        std::memcpy(sizes_arr.mutable_data(), region_sizes.data(), num * sizeof(long long));
        std::memcpy(seeds_arr.mutable_data(), seed_points.data(), num * 3 * sizeof(long long));
    }
    py::dict out;
    out["sizes"] = sizes_arr;
    out["seeds"] = seeds_arr;
    out["iterations"] = iteration;
    out["max_iterations_reached"] = (iteration >= max_iterations && !remaining.empty());
    out["remaining_voxels"] = (long long)remaining.size();
    return out;
}
