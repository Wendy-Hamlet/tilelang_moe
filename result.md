2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, k % 2 * 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            with T.attr(0, "async_wait_queue_scope", 0):
                T.attr(0, "async_wait_inflight_count", 0)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            for i in T.unroll(128):
                if i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and (i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start) // 128 + bx < 1:
                    if 0 <= bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start:
                        Y_1 = T.Buffer((49152,), "float16", data=Y.data)
                        Y_1[T.Cast("int64", bx) * T.int64(49152) + T.Cast("int64", i) // T.int64(64) * T.int64(24576) + T.Cast("int64", tx) // T.int64(32) * T.int64(6144) + T.Cast("int64", i) % T.int64(2) * T.int64(3072) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(384) + T.Cast("int64", expert_start) * T.int64(384) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2):T.Cast("int64", bx) * T.int64(49152) + T.Cast("int64", i) // T.int64(64) * T.int64(24576) + T.Cast("int64", tx) // T.int64(32) * T.int64(6144) + T.Cast("int64", i) % T.int64(2) * T.int64(3072) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(384) + T.Cast("int64", expert_start) * T.int64(384) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2) + T.int64(2)] = T.Cast("float16x2", C_local_1[i * 2:i * 2 + 2])
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   1.0911 ms
  pytorch-compiled         :   1.0964 ms
  tilelang-separate        :   1.2594 ms
2026-01-26 21:43:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
  ours (sonic)             : FAILED (Memory verification failed with the following errors:
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  Did you forget to bind?
# from tvm.script import tir as T

@T.prim_func
def grouped_gemm_sonic(X_handle: T.handle, W_handle: T.handle, expert_offsets_handle: T.handle, Y_handle: T.handle):
    T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    X = T.match_buffer(X_handle, (128, 512), "float16", strides=(512, 1))
    W = T.match_buffer(W_handle, (8, 384, 512), "float16", strides=(196608, 512, 1))
    expert_offsets = T.match_buffer(expert_offsets_handle, (9,), "int32", strides=(1,))
    Y = T.match_buffer(Y_handle, (128, 384), "float16", strides=(384, 1))
    for expert_id in range(8):
        expert_offsets_1 = T.Buffer((9,), "int32", data=expert_offsets.data)
        expert_start: T.int32 = expert_offsets_1[expert_id]
        expert_end: T.int32 = expert_offsets_1[expert_id + 1]
        if 0 < expert_end - expert_start:
            bx = T.launch_thread("blockIdx.x", (expert_end + 127 - expert_start) // 128)
            C_local = T.allocate([256], "float32", "local")
            X_shared = T.allocate([8192], "float16", "shared.dyn")
            W_shared = T.allocate([16384], "float16", "shared.dyn")
            desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
            desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
            by = T.launch_thread("blockIdx.y", 1)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            C_local_1 = T.Buffer((256,), data=C_local, scope="local")
            for i in T.unroll(128):
                C_local_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
            X_shared_1 = T.Buffer((8192,), "float16", data=X_shared, scope="shared.dyn")
            X_1 = T.Buffer((65536,), "float16", data=X.data)
            with T.attr(0, "async_scope", 1):
                for i in T.unroll(4):
                    if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and (i * 32 + tx // 4 + expert_start) // 128 + bx < 1:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(65536) + T.Cast("int64", i) * T.int64(16384) + T.Cast("int64", tx) // T.int64(4) * T.int64(512) + T.Cast("int64", expert_start) * T.int64(512) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", bx) * T.int64(65536) + T.Cast("int64", i) * T.int64(16384) + T.Cast("int64", tx) // T.int64(4) * T.int64(512) + T.Cast("int64", expert_start) * T.int64(512) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                    else:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
            W_shared_1 = T.Buffer((16384,), "float16", data=W_shared, scope="shared.dyn")
            W_1 = T.Buffer((1572864,), "float16", data=W.data)
            with T.attr(0, "async_commit_queue_scope", 0):
                T.attr(0, "async_scope", 1)
                for i in T.unroll(8):
                    W_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 196608 + i * 16384 + tx // 4 * 512 + tx % 4 * 8:expert_id * 196608 + i * 16384 + tx // 4 * 512 + tx % 4 * 8 + 8]
            desc_a_1 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
            desc_b_1 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
            for k in range(15):
                with T.attr(0, "async_scope", 1):
                    for i in T.unroll(4):
                        if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and (i * 32 + tx // 4 + expert_start) // 128 + bx < 1:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(65536) + T.Cast("int64", i) * T.int64(16384) + T.Cast("int64", tx) // T.int64(4) * T.int64(512) + T.Cast("int64", expert_start) * T.int64(512) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", bx) * T.int64(65536) + T.Cast("int64", i) * T.int64(16384) + T.Cast("int64", tx) // T.int64(4) * T.int64(512) + T.Cast("int64", expert_start) * T.int64(512) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                        else:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
                with T.attr(0, "async_commit_queue_scope", 0):
                    T.attr(0, "async_scope", 1)
                    for i in T.unroll(8):
                        W_shared_1[(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 196608 + i * 16384 + tx // 4 * 512 + k * 32 + tx % 4 * 8 + 32:expert_id * 196608 + i * 16384 + tx // 4 * 512 + k * 32 + tx % 4 * 8 + 32 + 8]
                T.attr(0, "async_wait_queue_scope", 0)
                T.attr(0, "async_wait_inflight_count", 1)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, k % 2 * 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, k % 2 * 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            with T.attr(0, "async_wait_queue_scope", 0):
                T.attr(0, "async_wait_inflight_count", 0)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            for i in T.unroll(128):
                if i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and (i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start) // 128 + bx < 1:
                    if 0 <= bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start:
                        Y_1 = T.Buffer((49152,), "float16", data=Y.data)
                        Y_1[T.Cast("int64", bx) * T.int64(49152) + T.Cast("int64", i) // T.int64(64) * T.int64(24576) + T.Cast("int64", tx) // T.int64(32) * T.int64(6144) + T.Cast("int64", i) % T.int64(2) * T.int64(3072) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(384) + T.Cast("int64", expert_start) * T.int64(384) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2):T.Cast("int64", bx) * T.int64(49152) + T.Cast("int64", i) // T.int64(64) * T.int64(24576) + T.Cast("int64", tx) // T.int64(32) * T.int64(6144) + T.Cast("int64", i) % T.int64(2) * T.int64(3072) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(384) + T.Cast("int64", expert_start) * T.int64(384) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2) + T.int64(2)] = T.Cast("float16x2", C_local_1[i * 2:i * 2 + 2]))

================================================================================
Config: B256_H1024_I768_E16 - Medium Test
  total_tokens=256, H=1024, I=768, E=16
================================================================================
Expert distribution: min=3, max=73, mean=16.0

[1] Correctness Verification:
2026-01-26 21:43:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
❌ Error in ours (sonic): Memory verification failed with the following errors:
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  Did you forget to bind?
# from tvm.script import tir as T

@T.prim_func
def grouped_gemm_sonic(X_handle: T.handle, W_handle: T.handle, expert_offsets_handle: T.handle, Y_handle: T.handle):
    T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    X = T.match_buffer(X_handle, (256, 1024), "float16", strides=(1024, 1))
    W = T.match_buffer(W_handle, (16, 768, 1024), "float16", strides=(786432, 1024, 1))
    expert_offsets = T.match_buffer(expert_offsets_handle, (17,), "int32", strides=(1,))
    Y = T.match_buffer(Y_handle, (256, 768), "float16", strides=(768, 1))
    for expert_id in range(16):
        expert_offsets_1 = T.Buffer((17,), "int32", data=expert_offsets.data)
        expert_start: T.int32 = expert_offsets_1[expert_id]
        expert_end: T.int32 = expert_offsets_1[expert_id + 1]
        if 0 < expert_end - expert_start:
            bx = T.launch_thread("blockIdx.x", (expert_end + 127 - expert_start) // 128)
            C_local = T.allocate([256], "float32", "local")
            X_shared = T.allocate([8192], "float16", "shared.dyn")
            W_shared = T.allocate([16384], "float16", "shared.dyn")
            desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
            desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
            by = T.launch_thread("blockIdx.y", 1)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            C_local_1 = T.Buffer((256,), data=C_local, scope="local")
            for i in T.unroll(128):
                C_local_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
            X_shared_1 = T.Buffer((8192,), "float16", data=X_shared, scope="shared.dyn")
            X_1 = T.Buffer((262144,), "float16", data=X.data)
            with T.attr(0, "async_scope", 1):
                for i in T.unroll(4):
                    if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i * 32 + tx // 4 + expert_start < 256:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(131072) + T.Cast("int64", i) * T.int64(32768) + T.Cast("int64", tx) // T.int64(4) * T.int64(1024) + T.Cast("int64", expert_start) * T.int64(1024) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", bx) * T.int64(131072) + T.Cast("int64", i) * T.int64(32768) + T.Cast("int64", tx) // T.int64(4) * T.int64(1024) + T.Cast("int64", expert_start) * T.int64(1024) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                    else:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
            W_shared_1 = T.Buffer((16384,), "float16", data=W_shared, scope="shared.dyn")
            W_1 = T.Buffer((12582912,), "float16", data=W.data)
            with T.attr(0, "async_commit_queue_scope", 0):
                T.attr(0, "async_scope", 1)
                for i in T.unroll(8):
                    W_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 786432 + i * 32768 + tx // 4 * 1024 + tx % 4 * 8:expert_id * 786432 + i * 32768 + tx // 4 * 1024 + tx % 4 * 8 + 8]
            desc_a_1 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
            desc_b_1 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
            for k in range(31):
                with T.attr(0, "async_scope", 1):
                    for i in T.unroll(4):
                        if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i * 32 + tx // 4 + expert_start < 256:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(131072) + T.Cast("int64", i) * T.int64(32768) + T.Cast("int64", tx) // T.int64(4) * T.int64(1024) + T.Cast("int64", expert_start) * T.int64(1024) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", bx) * T.int64(131072) + T.Cast("int64", i) * T.int64(32768) + T.Cast("int64", tx) // T.int64(4) * T.int64(1024) + T.Cast("int64", expert_start) * T.int64(1024) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                        else:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
                with T.attr(0, "async_commit_queue_scope", 0):
                    T.attr(0, "async_scope", 1)
                    for i in T.unroll(8):
                        W_shared_1[(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 786432 + i * 32768 + tx // 4 * 1024 + k * 32 + tx % 4 * 8 + 32:expert_id * 786432 + i * 32768 + tx // 4 * 1024 + k * 32 + tx % 4 * 8 + 32 + 8]
                T.attr(0, "async_wait_queue_scope", 0)
                T.attr(0, "async_wait_inflight_count", 1)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, k % 2 * 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, k % 2 * 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            with T.attr(0, "async_wait_queue_scope", 0):
                T.attr(0, "async_wait_inflight_count", 0)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            for i in T.unroll(128):
                if i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start < 256:
                    if 0 <= bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start:
                        Y_1 = T.Buffer((196608,), "float16", data=Y.data)
                        Y_1[T.Cast("int64", bx) * T.int64(98304) + T.Cast("int64", i) // T.int64(64) * T.int64(49152) + T.Cast("int64", tx) // T.int64(32) * T.int64(12288) + T.Cast("int64", i) % T.int64(2) * T.int64(6144) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(768) + T.Cast("int64", expert_start) * T.int64(768) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2):T.Cast("int64", bx) * T.int64(98304) + T.Cast("int64", i) // T.int64(64) * T.int64(49152) + T.Cast("int64", tx) // T.int64(32) * T.int64(12288) + T.Cast("int64", i) % T.int64(2) * T.int64(6144) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(768) + T.Cast("int64", expert_start) * T.int64(768) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2) + T.int64(2)] = T.Cast("float16x2", C_local_1[i * 2:i * 2 + 2])
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   2.1813 ms
  pytorch-compiled         :   2.1967 ms
  tilelang-separate        :   2.5464 ms
2026-01-26 21:43:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
  ours (sonic)             : FAILED (Memory verification failed with the following errors:
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  Did you forget to bind?
# from tvm.script import tir as T

@T.prim_func
def grouped_gemm_sonic(X_handle: T.handle, W_handle: T.handle, expert_offsets_handle: T.handle, Y_handle: T.handle):
    T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    X = T.match_buffer(X_handle, (256, 1024), "float16", strides=(1024, 1))
    W = T.match_buffer(W_handle, (16, 768, 1024), "float16", strides=(786432, 1024, 1))
    expert_offsets = T.match_buffer(expert_offsets_handle, (17,), "int32", strides=(1,))
    Y = T.match_buffer(Y_handle, (256, 768), "float16", strides=(768, 1))
    for expert_id in range(16):
        expert_offsets_1 = T.Buffer((17,), "int32", data=expert_offsets.data)
        expert_start: T.int32 = expert_offsets_1[expert_id]
        expert_end: T.int32 = expert_offsets_1[expert_id + 1]
        if 0 < expert_end - expert_start:
            bx = T.launch_thread("blockIdx.x", (expert_end + 127 - expert_start) // 128)
            C_local = T.allocate([256], "float32", "local")
            X_shared = T.allocate([8192], "float16", "shared.dyn")
            W_shared = T.allocate([16384], "float16", "shared.dyn")
            desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
            desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
            by = T.launch_thread("blockIdx.y", 1)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            C_local_1 = T.Buffer((256,), data=C_local, scope="local")
            for i in T.unroll(128):
                C_local_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
            X_shared_1 = T.Buffer((8192,), "float16", data=X_shared, scope="shared.dyn")
            X_1 = T.Buffer((262144,), "float16", data=X.data)
            with T.attr(0, "async_scope", 1):
                for i in T.unroll(4):
                    if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i * 32 + tx // 4 + expert_start < 256:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(131072) + T.Cast("int64", i) * T.int64(32768) + T.Cast("int64", tx) // T.int64(4) * T.int64(1024) + T.Cast("int64", expert_start) * T.int64(1024) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", bx) * T.int64(131072) + T.Cast("int64", i) * T.int64(32768) + T.Cast("int64", tx) // T.int64(4) * T.int64(1024) + T.Cast("int64", expert_start) * T.int64(1024) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                    else:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
            W_shared_1 = T.Buffer((16384,), "float16", data=W_shared, scope="shared.dyn")
            W_1 = T.Buffer((12582912,), "float16", data=W.data)
            with T.attr(0, "async_commit_queue_scope", 0):
                T.attr(0, "async_scope", 1)
                for i in T.unroll(8):
                    W_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 786432 + i * 32768 + tx // 4 * 1024 + tx % 4 * 8:expert_id * 786432 + i * 32768 + tx // 4 * 1024 + tx % 4 * 8 + 8]
            desc_a_1 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
            desc_b_1 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
            for k in range(31):
                with T.attr(0, "async_scope", 1):
                    for i in T.unroll(4):
                        if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i * 32 + tx // 4 + expert_start < 256:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(131072) + T.Cast("int64", i) * T.int64(32768) + T.Cast("int64", tx) // T.int64(4) * T.int64(1024) + T.Cast("int64", expert_start) * T.int64(1024) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", bx) * T.int64(131072) + T.Cast("int64", i) * T.int64(32768) + T.Cast("int64", tx) // T.int64(4) * T.int64(1024) + T.Cast("int64", expert_start) * T.int64(1024) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                        else:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
                with T.attr(0, "async_commit_queue_scope", 0):
                    T.attr(0, "async_scope", 1)
                    for i in T.unroll(8):
                        W_shared_1[(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 786432 + i * 32768 + tx // 4 * 1024 + k * 32 + tx % 4 * 8 + 32:expert_id * 786432 + i * 32768 + tx // 4 * 1024 + k * 32 + tx % 4 * 8 + 32 + 8]
                T.attr(0, "async_wait_queue_scope", 0)
                T.attr(0, "async_wait_inflight_count", 1)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, k % 2 * 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, k % 2 * 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            with T.attr(0, "async_wait_queue_scope", 0):
                T.attr(0, "async_wait_inflight_count", 0)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            for i in T.unroll(128):
                if i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start < 256:
                    if 0 <= bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start:
                        Y_1 = T.Buffer((196608,), "float16", data=Y.data)
                        Y_1[T.Cast("int64", bx) * T.int64(98304) + T.Cast("int64", i) // T.int64(64) * T.int64(49152) + T.Cast("int64", tx) // T.int64(32) * T.int64(12288) + T.Cast("int64", i) % T.int64(2) * T.int64(6144) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(768) + T.Cast("int64", expert_start) * T.int64(768) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2):T.Cast("int64", bx) * T.int64(98304) + T.Cast("int64", i) // T.int64(64) * T.int64(49152) + T.Cast("int64", tx) // T.int64(32) * T.int64(12288) + T.Cast("int64", i) % T.int64(2) * T.int64(6144) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(768) + T.Cast("int64", expert_start) * T.int64(768) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2) + T.int64(2)] = T.Cast("float16x2", C_local_1[i * 2:i * 2 + 2]))

================================================================================
Config: B512_H2048_I1536_E32 - DeepSeek-V2-like
  total_tokens=512, H=2048, I=1536, E=32
================================================================================
Expert distribution: min=1, max=127, mean=16.0

[1] Correctness Verification:
2026-01-26 21:43:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
❌ Error in ours (sonic): Memory verification failed with the following errors:
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  Did you forget to bind?
# from tvm.script import tir as T

@T.prim_func
def grouped_gemm_sonic(X_handle: T.handle, W_handle: T.handle, expert_offsets_handle: T.handle, Y_handle: T.handle):
    T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    X = T.match_buffer(X_handle, (512, 2048), "float16", strides=(2048, 1))
    W = T.match_buffer(W_handle, (32, 1536, 2048), "float16", strides=(3145728, 2048, 1))
    expert_offsets = T.match_buffer(expert_offsets_handle, (33,), "int32", strides=(1,))
    Y = T.match_buffer(Y_handle, (512, 1536), "float16", strides=(1536, 1))
    for expert_id in range(32):
        expert_offsets_1 = T.Buffer((33,), "int32", data=expert_offsets.data)
        expert_start: T.int32 = expert_offsets_1[expert_id]
        expert_end: T.int32 = expert_offsets_1[expert_id + 1]
        if 0 < expert_end - expert_start:
            bx = T.launch_thread("blockIdx.x", (expert_end + 127 - expert_start) // 128)
            C_local = T.allocate([256], "float32", "local")
            X_shared = T.allocate([8192], "float16", "shared.dyn")
            W_shared = T.allocate([16384], "float16", "shared.dyn")
            desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
            desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
            by = T.launch_thread("blockIdx.y", 1)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            C_local_1 = T.Buffer((256,), data=C_local, scope="local")
            for i in T.unroll(128):
                C_local_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
            X_shared_1 = T.Buffer((8192,), "float16", data=X_shared, scope="shared.dyn")
            X_1 = T.Buffer((1048576,), "float16", data=X.data)
            with T.attr(0, "async_scope", 1):
                for i in T.unroll(4):
                    if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i * 32 + tx // 4 + expert_start < 512:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(262144) + T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", expert_start) * T.int64(2048) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", bx) * T.int64(262144) + T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", expert_start) * T.int64(2048) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                    else:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
            W_shared_1 = T.Buffer((16384,), "float16", data=W_shared, scope="shared.dyn")
            W_1 = T.Buffer((100663296,), "float16", data=W.data)
            with T.attr(0, "async_commit_queue_scope", 0):
                T.attr(0, "async_scope", 1)
                for i in T.unroll(8):
                    W_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 3145728 + i * 65536 + tx // 4 * 2048 + tx % 4 * 8:expert_id * 3145728 + i * 65536 + tx // 4 * 2048 + tx % 4 * 8 + 8]
            desc_a_1 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
            desc_b_1 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
            for k in range(63):
                with T.attr(0, "async_scope", 1):
                    for i in T.unroll(4):
                        if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i * 32 + tx // 4 + expert_start < 512:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(262144) + T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", expert_start) * T.int64(2048) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", bx) * T.int64(262144) + T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", expert_start) * T.int64(2048) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                        else:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
                with T.attr(0, "async_commit_queue_scope", 0):
                    T.attr(0, "async_scope", 1)
                    for i in T.unroll(8):
                        W_shared_1[(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 3145728 + i * 65536 + tx // 4 * 2048 + k * 32 + tx % 4 * 8 + 32:expert_id * 3145728 + i * 65536 + tx // 4 * 2048 + k * 32 + tx % 4 * 8 + 32 + 8]
                T.attr(0, "async_wait_queue_scope", 0)
                T.attr(0, "async_wait_inflight_count", 1)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, k % 2 * 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, k % 2 * 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            with T.attr(0, "async_wait_queue_scope", 0):
                T.attr(0, "async_wait_inflight_count", 0)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            for i in T.unroll(128):
                if i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start < 512:
                    if 0 <= bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start:
                        Y_1 = T.Buffer((786432,), "float16", data=Y.data)
                        Y_1[T.Cast("int64", bx) * T.int64(196608) + T.Cast("int64", i) // T.int64(64) * T.int64(98304) + T.Cast("int64", tx) // T.int64(32) * T.int64(24576) + T.Cast("int64", i) % T.int64(2) * T.int64(12288) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(1536) + T.Cast("int64", expert_start) * T.int64(1536) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2):T.Cast("int64", bx) * T.int64(196608) + T.Cast("int64", i) // T.int64(64) * T.int64(98304) + T.Cast("int64", tx) // T.int64(32) * T.int64(24576) + T.Cast("int64", i) % T.int64(2) * T.int64(12288) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(1536) + T.Cast("int64", expert_start) * T.int64(1536) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2) + T.int64(2)] = T.Cast("float16x2", C_local_1[i * 2:i * 2 + 2])
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   4.4526 ms
  pytorch-compiled         :   5.3063 ms
  tilelang-separate        :   6.5251 ms
2026-01-26 21:43:16  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
  ours (sonic)             : FAILED (Memory verification failed with the following errors:
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  Did you forget to bind?
# from tvm.script import tir as T

@T.prim_func
def grouped_gemm_sonic(X_handle: T.handle, W_handle: T.handle, expert_offsets_handle: T.handle, Y_handle: T.handle):
    T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    X = T.match_buffer(X_handle, (512, 2048), "float16", strides=(2048, 1))
    W = T.match_buffer(W_handle, (32, 1536, 2048), "float16", strides=(3145728, 2048, 1))
    expert_offsets = T.match_buffer(expert_offsets_handle, (33,), "int32", strides=(1,))
    Y = T.match_buffer(Y_handle, (512, 1536), "float16", strides=(1536, 1))
    for expert_id in range(32):
        expert_offsets_1 = T.Buffer((33,), "int32", data=expert_offsets.data)
        expert_start: T.int32 = expert_offsets_1[expert_id]
        expert_end: T.int32 = expert_offsets_1[expert_id + 1]
        if 0 < expert_end - expert_start:
            bx = T.launch_thread("blockIdx.x", (expert_end + 127 - expert_start) // 128)
            C_local = T.allocate([256], "float32", "local")
            X_shared = T.allocate([8192], "float16", "shared.dyn")
            W_shared = T.allocate([16384], "float16", "shared.dyn")
            desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
            desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
            by = T.launch_thread("blockIdx.y", 1)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            C_local_1 = T.Buffer((256,), data=C_local, scope="local")
            for i in T.unroll(128):
                C_local_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
            X_shared_1 = T.Buffer((8192,), "float16", data=X_shared, scope="shared.dyn")
            X_1 = T.Buffer((1048576,), "float16", data=X.data)
            with T.attr(0, "async_scope", 1):
                for i in T.unroll(4):
                    if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i * 32 + tx // 4 + expert_start < 512:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(262144) + T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", expert_start) * T.int64(2048) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", bx) * T.int64(262144) + T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", expert_start) * T.int64(2048) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                    else:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
            W_shared_1 = T.Buffer((16384,), "float16", data=W_shared, scope="shared.dyn")
            W_1 = T.Buffer((100663296,), "float16", data=W.data)
            with T.attr(0, "async_commit_queue_scope", 0):
                T.attr(0, "async_scope", 1)
                for i in T.unroll(8):
                    W_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 3145728 + i * 65536 + tx // 4 * 2048 + tx % 4 * 8:expert_id * 3145728 + i * 65536 + tx // 4 * 2048 + tx % 4 * 8 + 8]
            desc_a_1 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
            desc_b_1 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
            for k in range(63):
                with T.attr(0, "async_scope", 1):
                    for i in T.unroll(4):
                        if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i * 32 + tx // 4 + expert_start < 512:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(262144) + T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", expert_start) * T.int64(2048) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", bx) * T.int64(262144) + T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", expert_start) * T.int64(2048) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                        else:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
                with T.attr(0, "async_commit_queue_scope", 0):
                    T.attr(0, "async_scope", 1)
                    for i in T.unroll(8):
                        W_shared_1[(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 3145728 + i * 65536 + tx // 4 * 2048 + k * 32 + tx % 4 * 8 + 32:expert_id * 3145728 + i * 65536 + tx // 4 * 2048 + k * 32 + tx % 4 * 8 + 32 + 8]
                T.attr(0, "async_wait_queue_scope", 0)
                T.attr(0, "async_wait_inflight_count", 1)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, k % 2 * 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, k % 2 * 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            with T.attr(0, "async_wait_queue_scope", 0):
                T.attr(0, "async_wait_inflight_count", 0)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            for i in T.unroll(128):
                if i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start < 512:
                    if 0 <= bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start:
                        Y_1 = T.Buffer((786432,), "float16", data=Y.data)
                        Y_1[T.Cast("int64", bx) * T.int64(196608) + T.Cast("int64", i) // T.int64(64) * T.int64(98304) + T.Cast("int64", tx) // T.int64(32) * T.int64(24576) + T.Cast("int64", i) % T.int64(2) * T.int64(12288) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(1536) + T.Cast("int64", expert_start) * T.int64(1536) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2):T.Cast("int64", bx) * T.int64(196608) + T.Cast("int64", i) // T.int64(64) * T.int64(98304) + T.Cast("int64", tx) // T.int64(32) * T.int64(24576) + T.Cast("int64", i) % T.int64(2) * T.int64(12288) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(1536) + T.Cast("int64", expert_start) * T.int64(1536) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2) + T.int64(2)] = T.Cast("float16x2", C_local_1[i * 2:i * 2 + 2]))

================================================================================
Config: B1024_H4096_I3072_E64 - DeepSeek-V3-like (scaled)
  total_tokens=1024, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=208, mean=16.0

[1] Correctness Verification:
2026-01-26 21:43:16  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
❌ Error in ours (sonic): Memory verification failed with the following errors:
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  Did you forget to bind?
# from tvm.script import tir as T

@T.prim_func
def grouped_gemm_sonic(X_handle: T.handle, W_handle: T.handle, expert_offsets_handle: T.handle, Y_handle: T.handle):
    T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    X = T.match_buffer(X_handle, (1024, 4096), "float16", strides=(4096, 1))
    W = T.match_buffer(W_handle, (64, 3072, 4096), "float16", strides=(12582912, 4096, 1))
    expert_offsets = T.match_buffer(expert_offsets_handle, (65,), "int32", strides=(1,))
    Y = T.match_buffer(Y_handle, (1024, 3072), "float16", strides=(3072, 1))
    for expert_id in range(64):
        expert_offsets_1 = T.Buffer((65,), "int32", data=expert_offsets.data)
        expert_start: T.int32 = expert_offsets_1[expert_id]
        expert_end: T.int32 = expert_offsets_1[expert_id + 1]
        if 0 < expert_end - expert_start:
            bx = T.launch_thread("blockIdx.x", (expert_end + 127 - expert_start) // 128)
            C_local = T.allocate([256], "float32", "local")
            X_shared = T.allocate([8192], "float16", "shared.dyn")
            W_shared = T.allocate([16384], "float16", "shared.dyn")
            desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
            desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
            by = T.launch_thread("blockIdx.y", 1)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            C_local_1 = T.Buffer((256,), data=C_local, scope="local")
            for i in T.unroll(128):
                C_local_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
            X_shared_1 = T.Buffer((8192,), "float16", data=X_shared, scope="shared.dyn")
            X_1 = T.Buffer((4194304,), "float16", data=X.data)
            with T.attr(0, "async_scope", 1):
                for i in T.unroll(4):
                    if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i * 32 + tx // 4 + expert_start < 1024:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(524288) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", bx) * T.int64(524288) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                    else:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
            W_shared_1 = T.Buffer((16384,), "float16", data=W_shared, scope="shared.dyn")
            W_1 = T.Buffer((805306368,), "float16", data=W.data)
            with T.attr(0, "async_commit_queue_scope", 0):
                T.attr(0, "async_scope", 1)
                for i in T.unroll(8):
                    W_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 12582912 + i * 131072 + tx // 4 * 4096 + tx % 4 * 8:expert_id * 12582912 + i * 131072 + tx // 4 * 4096 + tx % 4 * 8 + 8]
            desc_a_1 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
            desc_b_1 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
            for k in range(127):
                with T.attr(0, "async_scope", 1):
                    for i in T.unroll(4):
                        if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i * 32 + tx // 4 + expert_start < 1024:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(524288) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", bx) * T.int64(524288) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                        else:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
                with T.attr(0, "async_commit_queue_scope", 0):
                    T.attr(0, "async_scope", 1)
                    for i in T.unroll(8):
                        W_shared_1[(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 12582912 + i * 131072 + tx // 4 * 4096 + k * 32 + tx % 4 * 8 + 32:expert_id * 12582912 + i * 131072 + tx // 4 * 4096 + k * 32 + tx % 4 * 8 + 32 + 8]
                T.attr(0, "async_wait_queue_scope", 0)
                T.attr(0, "async_wait_inflight_count", 1)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, k % 2 * 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, k % 2 * 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            with T.attr(0, "async_wait_queue_scope", 0):
                T.attr(0, "async_wait_inflight_count", 0)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            for i in T.unroll(128):
                if i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start < 1024:
                    if 0 <= bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start:
                        Y_1 = T.Buffer((3145728,), "float16", data=Y.data)
                        Y_1[T.Cast("int64", bx) * T.int64(393216) + T.Cast("int64", i) // T.int64(64) * T.int64(196608) + T.Cast("int64", tx) // T.int64(32) * T.int64(49152) + T.Cast("int64", i) % T.int64(2) * T.int64(24576) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(3072) + T.Cast("int64", expert_start) * T.int64(3072) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2):T.Cast("int64", bx) * T.int64(393216) + T.Cast("int64", i) // T.int64(64) * T.int64(196608) + T.Cast("int64", tx) // T.int64(32) * T.int64(49152) + T.Cast("int64", i) % T.int64(2) * T.int64(24576) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(3072) + T.Cast("int64", expert_start) * T.int64(3072) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2) + T.int64(2)] = T.Cast("float16x2", C_local_1[i * 2:i * 2 + 2])
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.250000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :  10.9931 ms
  pytorch-compiled         :  11.0114 ms
  tilelang-separate        :  14.6909 ms
2026-01-26 21:43:18  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
  ours (sonic)             : FAILED (Memory verification failed with the following errors:
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
    Variable `expert_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  Did you forget to bind?
# from tvm.script import tir as T

@T.prim_func
def grouped_gemm_sonic(X_handle: T.handle, W_handle: T.handle, expert_offsets_handle: T.handle, Y_handle: T.handle):
    T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    X = T.match_buffer(X_handle, (1024, 4096), "float16", strides=(4096, 1))
    W = T.match_buffer(W_handle, (64, 3072, 4096), "float16", strides=(12582912, 4096, 1))
    expert_offsets = T.match_buffer(expert_offsets_handle, (65,), "int32", strides=(1,))
    Y = T.match_buffer(Y_handle, (1024, 3072), "float16", strides=(3072, 1))
    for expert_id in range(64):
        expert_offsets_1 = T.Buffer((65,), "int32", data=expert_offsets.data)
        expert_start: T.int32 = expert_offsets_1[expert_id]
        expert_end: T.int32 = expert_offsets_1[expert_id + 1]
        if 0 < expert_end - expert_start:
            bx = T.launch_thread("blockIdx.x", (expert_end + 127 - expert_start) // 128)
            C_local = T.allocate([256], "float32", "local")
            X_shared = T.allocate([8192], "float16", "shared.dyn")
            W_shared = T.allocate([16384], "float16", "shared.dyn")
            desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
            desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
            by = T.launch_thread("blockIdx.y", 1)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            C_local_1 = T.Buffer((256,), data=C_local, scope="local")
            for i in T.unroll(128):
                C_local_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
            X_shared_1 = T.Buffer((8192,), "float16", data=X_shared, scope="shared.dyn")
            X_1 = T.Buffer((4194304,), "float16", data=X.data)
            with T.attr(0, "async_scope", 1):
                for i in T.unroll(4):
                    if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i * 32 + tx // 4 + expert_start < 1024:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(524288) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", bx) * T.int64(524288) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                    else:
                        X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
            W_shared_1 = T.Buffer((16384,), "float16", data=W_shared, scope="shared.dyn")
            W_1 = T.Buffer((805306368,), "float16", data=W.data)
            with T.attr(0, "async_commit_queue_scope", 0):
                T.attr(0, "async_scope", 1)
                for i in T.unroll(8):
                    W_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 12582912 + i * 131072 + tx // 4 * 4096 + tx % 4 * 8:expert_id * 12582912 + i * 131072 + tx // 4 * 4096 + tx % 4 * 8 + 8]
            desc_a_1 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
            desc_b_1 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
            for k in range(127):
                with T.attr(0, "async_scope", 1):
                    for i in T.unroll(4):
                        if i * 32 + tx // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i * 32 + tx // 4 + expert_start < 1024:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= bx * 128 + i * 32 + tx // 4 + expert_start, X_1[T.Cast("int64", bx) * T.int64(524288) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", bx) * T.int64(524288) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                        else:
                            X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
                with T.attr(0, "async_commit_queue_scope", 0):
                    T.attr(0, "async_scope", 1)
                    for i in T.unroll(8):
                        W_shared_1[(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = W_1[expert_id * 12582912 + i * 131072 + tx // 4 * 4096 + k * 32 + tx % 4 * 8 + 32:expert_id * 12582912 + i * 131072 + tx // 4 * 4096 + k * 32 + tx % 4 * 8 + 32 + 8]
                T.attr(0, "async_wait_queue_scope", 0)
                T.attr(0, "async_wait_inflight_count", 1)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, k % 2 * 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, k % 2 * 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            with T.attr(0, "async_wait_queue_scope", 0):
                T.attr(0, "async_wait_inflight_count", 0)
                T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), X_shared, 4096, 4096, 1), 2, 1, 32)
                T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), W_shared, 8192, 8192, 1), 2, 1, 32)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
                T.warpgroup_arrive()
                T.fence_proxy_async()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss("m64n256k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a, T.shift_right(i * 4096 + ki * 32, 4), desc_b, T.shift_right(ki * 32, 4), C_local, i * 128, 1, 1, 1)
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 256)
            for i in T.unroll(128):
                if i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 < T.min(bx * 128 + 128, expert_end - expert_start) - bx * 128 and bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start < 1024:
                    if 0 <= bx * 128 + i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start:
                        Y_1 = T.Buffer((3145728,), "float16", data=Y.data)
                        Y_1[T.Cast("int64", bx) * T.int64(393216) + T.Cast("int64", i) // T.int64(64) * T.int64(196608) + T.Cast("int64", tx) // T.int64(32) * T.int64(49152) + T.Cast("int64", i) % T.int64(2) * T.int64(24576) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(3072) + T.Cast("int64", expert_start) * T.int64(3072) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2):T.Cast("int64", bx) * T.int64(393216) + T.Cast("int64", i) // T.int64(64) * T.int64(196608) + T.Cast("int64", tx) // T.int64(32) * T.int64(49152) + T.Cast("int64", i) % T.int64(2) * T.int64(24576) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(3072) + T.Cast("int64", expert_start) * T.int64(3072) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2) + T.int64(2)] = T.Cast("float16x2", C_local_1[i * 2:i * 2 + 2]))

================================================================================
SUMMARY: Latency Comparison (ms)
================================================================================
+-----------------------+-----------+------------+----------------+----------------+
| Config                |   PyTorch |   Compiled |   TileLang-Sep | Ours (Sonic)   |
+=======================+===========+============+================+================+
| B128_H512_I384_E8     |    1.0911 |     1.0964 |         1.2594 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B256_H1024_I768_E16   |    2.1813 |     2.1967 |         2.5464 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B512_H2048_I1536_E32  |    4.4526 |     5.3063 |         6.5251 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 |   10.9931 |    11.0114 |        14.6909 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+

================================================================================
SUMMARY: Speedup vs PyTorch Baseline
================================================================================
+-----------------------+------------+----------------+----------------+
| Config                | Compiled   | TileLang-Sep   | Ours (Sonic)   |
+=======================+============+================+================+
| B128_H512_I384_E8     | 1.00x      | 0.87x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B256_H1024_I768_E16   | 0.99x      | 0.86x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B512_H2048_I1536_E32  | 0.84x      | 0.68x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 | 1.00x      | 0.75x          | N/A            |
+-----------------------+------------+----------------+----------------+

================================================================================
Average Speedup:
================================================================================
  pytorch-compiled         : 0.96x
  tilelang-separate        : 0.79x