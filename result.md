[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   5.0178 ms
  pytorch-compiled         :   5.0274 ms
  tilelang-separate        :   6.2036 ms
2026-01-26 21:48:15  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
  ours (sonic)             : FAILED (Memory verification failed with the following errors:
    Variable `block_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  Did you forget to bind?
# from tvm.script import tir as T

@T.prim_func
def grouped_gemm_sonic(X_handle: T.handle, W_handle: T.handle, expert_offsets_handle: T.handle, block_offsets_handle: T.handle, Y_handle: T.handle):
    T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    X = T.match_buffer(X_handle, (512, 2048), "float16", strides=(2048, 1))
    W = T.match_buffer(W_handle, (32, 1536, 2048), "float16", strides=(3145728, 2048, 1))
    expert_offsets = T.match_buffer(expert_offsets_handle, (33,), "int32", strides=(1,))
    block_offsets = T.match_buffer(block_offsets_handle, (33,), "int32", strides=(1,))
    Y = T.match_buffer(Y_handle, (512, 1536), "float16", strides=(1536, 1))
    block_offsets_1 = T.Buffer((33,), "int32", data=block_offsets.data)
    total_blocks: T.int32 = block_offsets_1[32]
    bx = T.launch_thread("blockIdx.x", total_blocks)
    expert_id = T.allocate([1], "int32", "local.var")
    local_block_id = T.allocate([1], "int32", "local.var")
    C_local = T.allocate([256], "float32", "local")
    X_shared = T.allocate([8192], "float16", "shared.dyn")
    W_shared = T.allocate([16384], "float16", "shared.dyn")
    desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
    desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
    by = T.launch_thread("blockIdx.y", 1)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    expert_id_1 = T.Buffer((1,), "int32", data=expert_id, scope="local.var")
    expert_id_1[0] = 0
    local_block_id_1 = T.Buffer((1,), "int32", data=local_block_id, scope="local.var")
    local_block_id_1[0] = 0
    for e in range(32):
        if block_offsets_1[e] <= bx and bx < block_offsets_1[e + 1]:
            expert_id_1[0] = e
            local_block_id_1[0] = bx - block_offsets_1[e]
            T.break_loop()
    expert_offsets_1 = T.Buffer((33,), "int32", data=expert_offsets.data)
    expert_start: T.int32 = T.if_then_else(0 <= expert_id_1[0] and expert_id_1[0] < 33, expert_offsets_1[expert_id_1[T.int64(0)]], 0)
    expert_end: T.int32 = T.if_then_else(-1 <= expert_id_1[0] and expert_id_1[0] < 32, expert_offsets_1[T.Cast("int64", expert_id_1[T.int64(0)]) + T.int64(1)], 0)
    local_token_start: T.int32 = local_block_id_1[0] * 128
    if 0 < expert_end - expert_start - local_token_start:
        C_local_1 = T.Buffer((256,), data=C_local, scope="local")
        for i in T.unroll(128):
            C_local_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
        X_shared_1 = T.Buffer((8192,), "float16", data=X_shared, scope="shared.dyn")
        X_1 = T.Buffer((1048576,), "float16", data=X.data)
        with T.attr(0, "async_scope", 1):
            for i in T.unroll(4):
                if i * 32 + tx // 4 < expert_end - expert_start - local_token_start and i * 32 + tx // 4 + expert_start + local_token_start < 512:
                    X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= i * 32 + tx // 4 + expert_start + local_token_start, X_1[T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", expert_start) * T.int64(2048) + T.Cast("int64", local_token_start) * T.int64(2048) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", expert_start) * T.int64(2048) + T.Cast("int64", local_token_start) * T.int64(2048) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                else:
                    X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
        W_shared_1 = T.Buffer((16384,), "float16", data=W_shared, scope="shared.dyn")
        W_1 = T.Buffer((100663296,), "float16", data=W.data)
        with T.attr(0, "async_commit_queue_scope", 0):
            T.attr(0, "async_scope", 1)
            for i in T.unroll(8):
                W_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= expert_id_1[0] and expert_id_1[0] < 32, W_1[T.Cast("int64", expert_id_1[T.int64(0)]) * T.int64(3145728) + T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", expert_id_1[T.int64(0)]) * T.int64(3145728) + T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
        desc_a_1 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
        desc_b_1 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
        for k in range(63):
            with T.attr(0, "async_scope", 1):
                for i in T.unroll(4):
                    if i * 32 + tx // 4 < expert_end - expert_start - local_token_start and i * 32 + tx // 4 + expert_start + local_token_start < 512:
                        X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= i * 32 + tx // 4 + expert_start + local_token_start, X_1[T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", expert_start) * T.int64(2048) + T.Cast("int64", local_token_start) * T.int64(2048) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", expert_start) * T.int64(2048) + T.Cast("int64", local_token_start) * T.int64(2048) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                    else:
                        X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
            with T.attr(0, "async_commit_queue_scope", 0):
                T.attr(0, "async_scope", 1)
                for i in T.unroll(8):
                    W_shared_1[(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= expert_id_1[0] and expert_id_1[0] < 32, W_1[T.Cast("int64", expert_id_1[T.int64(0)]) * T.int64(3145728) + T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", expert_id_1[T.int64(0)]) * T.int64(3145728) + T.Cast("int64", i) * T.int64(65536) + T.Cast("int64", tx) // T.int64(4) * T.int64(2048) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
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
            if i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 < expert_end - expert_start - local_token_start and i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start + local_token_start < 512:
                if 0 <= i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start + local_token_start:
                    Y_1 = T.Buffer((786432,), "float16", data=Y.data)
                    Y_1[T.Cast("int64", i) // T.int64(64) * T.int64(98304) + T.Cast("int64", tx) // T.int64(32) * T.int64(24576) + T.Cast("int64", i) % T.int64(2) * T.int64(12288) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(1536) + T.Cast("int64", expert_start) * T.int64(1536) + T.Cast("int64", local_token_start) * T.int64(1536) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2):T.Cast("int64", i) // T.int64(64) * T.int64(98304) + T.Cast("int64", tx) // T.int64(32) * T.int64(24576) + T.Cast("int64", i) % T.int64(2) * T.int64(12288) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(1536) + T.Cast("int64", expert_start) * T.int64(1536) + T.Cast("int64", local_token_start) * T.int64(1536) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2) + T.int64(2)] = T.Cast("float16x2", C_local_1[i * 2:i * 2 + 2]))

================================================================================
Config: B1024_H4096_I3072_E64 - DeepSeek-V3-like (scaled)
  total_tokens=1024, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=208, mean=16.0

[1] Correctness Verification:
2026-01-26 21:48:15  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
❌ Error in ours (sonic): Memory verification failed with the following errors:
    Variable `block_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  Did you forget to bind?
# from tvm.script import tir as T

@T.prim_func
def grouped_gemm_sonic(X_handle: T.handle, W_handle: T.handle, expert_offsets_handle: T.handle, block_offsets_handle: T.handle, Y_handle: T.handle):
    T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    X = T.match_buffer(X_handle, (1024, 4096), "float16", strides=(4096, 1))
    W = T.match_buffer(W_handle, (64, 3072, 4096), "float16", strides=(12582912, 4096, 1))
    expert_offsets = T.match_buffer(expert_offsets_handle, (65,), "int32", strides=(1,))
    block_offsets = T.match_buffer(block_offsets_handle, (65,), "int32", strides=(1,))
    Y = T.match_buffer(Y_handle, (1024, 3072), "float16", strides=(3072, 1))
    block_offsets_1 = T.Buffer((65,), "int32", data=block_offsets.data)
    total_blocks: T.int32 = block_offsets_1[64]
    bx = T.launch_thread("blockIdx.x", total_blocks)
    expert_id = T.allocate([1], "int32", "local.var")
    local_block_id = T.allocate([1], "int32", "local.var")
    C_local = T.allocate([256], "float32", "local")
    X_shared = T.allocate([8192], "float16", "shared.dyn")
    W_shared = T.allocate([16384], "float16", "shared.dyn")
    desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
    desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
    by = T.launch_thread("blockIdx.y", 1)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    expert_id_1 = T.Buffer((1,), "int32", data=expert_id, scope="local.var")
    expert_id_1[0] = 0
    local_block_id_1 = T.Buffer((1,), "int32", data=local_block_id, scope="local.var")
    local_block_id_1[0] = 0
    for e in range(64):
        if block_offsets_1[e] <= bx and bx < block_offsets_1[e + 1]:
            expert_id_1[0] = e
            local_block_id_1[0] = bx - block_offsets_1[e]
            T.break_loop()
    expert_offsets_1 = T.Buffer((65,), "int32", data=expert_offsets.data)
    expert_start: T.int32 = T.if_then_else(0 <= expert_id_1[0] and expert_id_1[0] < 65, expert_offsets_1[expert_id_1[T.int64(0)]], 0)
    expert_end: T.int32 = T.if_then_else(-1 <= expert_id_1[0] and expert_id_1[0] < 64, expert_offsets_1[T.Cast("int64", expert_id_1[T.int64(0)]) + T.int64(1)], 0)
    local_token_start: T.int32 = local_block_id_1[0] * 128
    if 0 < expert_end - expert_start - local_token_start:
        C_local_1 = T.Buffer((256,), data=C_local, scope="local")
        for i in T.unroll(128):
            C_local_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
        X_shared_1 = T.Buffer((8192,), "float16", data=X_shared, scope="shared.dyn")
        X_1 = T.Buffer((4194304,), "float16", data=X.data)
        with T.attr(0, "async_scope", 1):
            for i in T.unroll(4):
                if i * 32 + tx // 4 < expert_end - expert_start - local_token_start and i * 32 + tx // 4 + expert_start + local_token_start < 1024:
                    X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= i * 32 + tx // 4 + expert_start + local_token_start, X_1[T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", local_token_start) * T.int64(4096) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", local_token_start) * T.int64(4096) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                else:
                    X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
        W_shared_1 = T.Buffer((16384,), "float16", data=W_shared, scope="shared.dyn")
        W_1 = T.Buffer((805306368,), "float16", data=W.data)
        with T.attr(0, "async_commit_queue_scope", 0):
            T.attr(0, "async_scope", 1)
            for i in T.unroll(8):
                W_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= expert_id_1[0] and expert_id_1[0] < 64, W_1[T.Cast("int64", expert_id_1[T.int64(0)]) * T.int64(12582912) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", expert_id_1[T.int64(0)]) * T.int64(12582912) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
        desc_a_1 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
        desc_b_1 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
        for k in range(127):
            with T.attr(0, "async_scope", 1):
                for i in T.unroll(4):
                    if i * 32 + tx // 4 < expert_end - expert_start - local_token_start and i * 32 + tx // 4 + expert_start + local_token_start < 1024:
                        X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= i * 32 + tx // 4 + expert_start + local_token_start, X_1[T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", local_token_start) * T.int64(4096) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", local_token_start) * T.int64(4096) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                    else:
                        X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
            with T.attr(0, "async_commit_queue_scope", 0):
                T.attr(0, "async_scope", 1)
                for i in T.unroll(8):
                    W_shared_1[(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= expert_id_1[0] and expert_id_1[0] < 64, W_1[T.Cast("int64", expert_id_1[T.int64(0)]) * T.int64(12582912) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", expert_id_1[T.int64(0)]) * T.int64(12582912) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
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
            if i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 < expert_end - expert_start - local_token_start and i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start + local_token_start < 1024:
                if 0 <= i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start + local_token_start:
                    Y_1 = T.Buffer((3145728,), "float16", data=Y.data)
                    Y_1[T.Cast("int64", i) // T.int64(64) * T.int64(196608) + T.Cast("int64", tx) // T.int64(32) * T.int64(49152) + T.Cast("int64", i) % T.int64(2) * T.int64(24576) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(3072) + T.Cast("int64", expert_start) * T.int64(3072) + T.Cast("int64", local_token_start) * T.int64(3072) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2):T.Cast("int64", i) // T.int64(64) * T.int64(196608) + T.Cast("int64", tx) // T.int64(32) * T.int64(49152) + T.Cast("int64", i) % T.int64(2) * T.int64(24576) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(3072) + T.Cast("int64", expert_start) * T.int64(3072) + T.Cast("int64", local_token_start) * T.int64(3072) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2) + T.int64(2)] = T.Cast("float16x2", C_local_1[i * 2:i * 2 + 2])
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.250000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :  10.1613 ms
  pytorch-compiled         :  10.1718 ms
  tilelang-separate        :  13.9916 ms
2026-01-26 21:48:18  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
  ours (sonic)             : FAILED (Memory verification failed with the following errors:
    Variable `block_offsets` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  Did you forget to bind?
# from tvm.script import tir as T

@T.prim_func
def grouped_gemm_sonic(X_handle: T.handle, W_handle: T.handle, expert_offsets_handle: T.handle, block_offsets_handle: T.handle, Y_handle: T.handle):
    T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    X = T.match_buffer(X_handle, (1024, 4096), "float16", strides=(4096, 1))
    W = T.match_buffer(W_handle, (64, 3072, 4096), "float16", strides=(12582912, 4096, 1))
    expert_offsets = T.match_buffer(expert_offsets_handle, (65,), "int32", strides=(1,))
    block_offsets = T.match_buffer(block_offsets_handle, (65,), "int32", strides=(1,))
    Y = T.match_buffer(Y_handle, (1024, 3072), "float16", strides=(3072, 1))
    block_offsets_1 = T.Buffer((65,), "int32", data=block_offsets.data)
    total_blocks: T.int32 = block_offsets_1[64]
    bx = T.launch_thread("blockIdx.x", total_blocks)
    expert_id = T.allocate([1], "int32", "local.var")
    local_block_id = T.allocate([1], "int32", "local.var")
    C_local = T.allocate([256], "float32", "local")
    X_shared = T.allocate([8192], "float16", "shared.dyn")
    W_shared = T.allocate([16384], "float16", "shared.dyn")
    desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
    desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
    by = T.launch_thread("blockIdx.y", 1)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    expert_id_1 = T.Buffer((1,), "int32", data=expert_id, scope="local.var")
    expert_id_1[0] = 0
    local_block_id_1 = T.Buffer((1,), "int32", data=local_block_id, scope="local.var")
    local_block_id_1[0] = 0
    for e in range(64):
        if block_offsets_1[e] <= bx and bx < block_offsets_1[e + 1]:
            expert_id_1[0] = e
            local_block_id_1[0] = bx - block_offsets_1[e]
            T.break_loop()
    expert_offsets_1 = T.Buffer((65,), "int32", data=expert_offsets.data)
    expert_start: T.int32 = T.if_then_else(0 <= expert_id_1[0] and expert_id_1[0] < 65, expert_offsets_1[expert_id_1[T.int64(0)]], 0)
    expert_end: T.int32 = T.if_then_else(-1 <= expert_id_1[0] and expert_id_1[0] < 64, expert_offsets_1[T.Cast("int64", expert_id_1[T.int64(0)]) + T.int64(1)], 0)
    local_token_start: T.int32 = local_block_id_1[0] * 128
    if 0 < expert_end - expert_start - local_token_start:
        C_local_1 = T.Buffer((256,), data=C_local, scope="local")
        for i in T.unroll(128):
            C_local_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
        X_shared_1 = T.Buffer((8192,), "float16", data=X_shared, scope="shared.dyn")
        X_1 = T.Buffer((4194304,), "float16", data=X.data)
        with T.attr(0, "async_scope", 1):
            for i in T.unroll(4):
                if i * 32 + tx // 4 < expert_end - expert_start - local_token_start and i * 32 + tx // 4 + expert_start + local_token_start < 1024:
                    X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= i * 32 + tx // 4 + expert_start + local_token_start, X_1[T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", local_token_start) * T.int64(4096) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", local_token_start) * T.int64(4096) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                else:
                    X_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
        W_shared_1 = T.Buffer((16384,), "float16", data=W_shared, scope="shared.dyn")
        W_1 = T.Buffer((805306368,), "float16", data=W.data)
        with T.attr(0, "async_commit_queue_scope", 0):
            T.attr(0, "async_scope", 1)
            for i in T.unroll(8):
                W_shared_1[i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= expert_id_1[0] and expert_id_1[0] < 64, W_1[T.Cast("int64", expert_id_1[T.int64(0)]) * T.int64(12582912) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", tx) % T.int64(4) * T.int64(8):T.Cast("int64", expert_id_1[T.int64(0)]) * T.int64(12582912) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
        desc_a_1 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
        desc_b_1 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
        for k in range(127):
            with T.attr(0, "async_scope", 1):
                for i in T.unroll(4):
                    if i * 32 + tx // 4 < expert_end - expert_start - local_token_start and i * 32 + tx // 4 + expert_start + local_token_start < 1024:
                        X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= i * 32 + tx // 4 + expert_start + local_token_start, X_1[T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", local_token_start) * T.int64(4096) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", expert_start) * T.int64(4096) + T.Cast("int64", local_token_start) * T.int64(4096) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
                    else:
                        X_shared_1[(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 4096 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.Broadcast(T.float16(0.0), 8)
            with T.attr(0, "async_commit_queue_scope", 0):
                T.attr(0, "async_scope", 1)
                for i in T.unroll(8):
                    W_shared_1[(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8:(k + 1) % 2 * 8192 + i * 1024 + tx // 4 * 32 + (tx % 32 // 16 + tx % 4 // 2) % 2 * 16 + (tx % 16 // 8 + tx % 2) % 2 * 8 + 8] = T.if_then_else(0 <= expert_id_1[0] and expert_id_1[0] < 64, W_1[T.Cast("int64", expert_id_1[T.int64(0)]) * T.int64(12582912) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32):T.Cast("int64", expert_id_1[T.int64(0)]) * T.int64(12582912) + T.Cast("int64", i) * T.int64(131072) + T.Cast("int64", tx) // T.int64(4) * T.int64(4096) + T.Cast("int64", k) * T.int64(32) + T.Cast("int64", tx) % T.int64(4) * T.int64(8) + T.int64(32) + T.int64(8)], T.Broadcast(T.float16(0.0), 8))
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
            if i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 < expert_end - expert_start - local_token_start and i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start + local_token_start < 1024:
                if 0 <= i // 64 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 + expert_start + local_token_start:
                    Y_1 = T.Buffer((3145728,), "float16", data=Y.data)
                    Y_1[T.Cast("int64", i) // T.int64(64) * T.int64(196608) + T.Cast("int64", tx) // T.int64(32) * T.int64(49152) + T.Cast("int64", i) % T.int64(2) * T.int64(24576) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(3072) + T.Cast("int64", expert_start) * T.int64(3072) + T.Cast("int64", local_token_start) * T.int64(3072) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2):T.Cast("int64", i) // T.int64(64) * T.int64(196608) + T.Cast("int64", tx) // T.int64(32) * T.int64(49152) + T.Cast("int64", i) % T.int64(2) * T.int64(24576) + T.Cast("int64", tx) % T.int64(32) // T.int64(4) * T.int64(3072) + T.Cast("int64", expert_start) * T.int64(3072) + T.Cast("int64", local_token_start) * T.int64(3072) + T.Cast("int64", i) % T.int64(64) // T.int64(2) * T.int64(8) + T.Cast("int64", tx) % T.int64(4) * T.int64(2) + T.int64(2)] = T.Cast("float16x2", C_local_1[i * 2:i * 2 + 2]))

================================================================================
SUMMARY: Latency Comparison (ms)
================================================================================
+-----------------------+-----------+------------+----------------+----------------+
| Config                |   PyTorch |   Compiled |   TileLang-Sep | Ours (Sonic)   |
+=======================+===========+============+================+================+
| B128_H512_I384_E8     |    1.2597 |     1.2653 |         1.4419 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B256_H1024_I768_E16   |    2.5068 |     2.5185 |         2.8615 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B512_H2048_I1536_E32  |    5.0178 |     5.0274 |         6.2036 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 |   10.1613 |    10.1718 |        13.9916 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+

================================================================================
SUMMARY: Speedup vs PyTorch Baseline
================================================================================
+-----------------------+------------+----------------+----------------+
| Config                | Compiled   | TileLang-Sep   | Ours (Sonic)   |
+=======================+============+================+================+
| B128_H512_I384_E8     | 1.00x      | 0.87x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B256_H1024_I768_E16   | 1.00x      | 0.88x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B512_H2048_I1536_E32  | 1.00x      | 0.81x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 | 1.00x      | 0.73x          | N/A            |
+-----------------------+------------+----------------+----------------+

================================================================================
Average Speedup:
================================================================================
  pytorch-compiled         : 1.00x
  tilelang-separate        : 0.82x