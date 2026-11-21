@torch.inference_mode()
def generate_probe(front, ids, *, temperature, graphs=False, **kwargs):
    from relicllm.models.deepseek_v4_1 import ascend_linear, cube_experiment, modules
    from relicllm.models.deepseek_v4_1.generate import Generation
    from relicllm.models.deepseek_v4_1.graphs import snapshot, restore
    from relic_core.kernels import native_cube_block_gemv as native

    if temperature != 0 or graphs:
        raise ValueError("the isolated Cube probe requires greedy eager CLI entry")
    if os.environ.get("POCKETLLM_V41_MMA_EMULATE") != "1":
        raise RuntimeError("the CPU/exact default must stay enabled in the Cube experiment")
    variant = os.environ.get("RELIC_CUBE_VARIANT", "block32")
    if variant not in {"block32", "scaled_bf16", "weight_quant", "decode", "lookup"}:
        raise ValueError("unknown isolated Cube projection variant")
    rank = dist.get_rank()
    model = front.model
    prefetch = None
    activation_binding = None
    if os.environ.get("RELIC_VECTOR_FP4_WIDEN") == "1":
        from relic_core.kernels import unpack, vector_fp4_decode
        unpack.unpack_fp4_to_int8 = partial(vector_fp4_decode.unpack_fp4_vector, mode="arithmetic")
    elif os.environ.get("RELIC_STATIC_FP4_WIDEN") == "1":
        from relic_core.kernels import unpack,static_fp4_decode
        unpack.unpack_fp4_to_int8 = static_fp4_decode.unpack_fp4_pairs
    if os.environ.get("RELIC_EXACT_CPU_POOL") == "1":
        from relic_core.kernels import cpu_raw_simd_gemv, cpu_raw_simd_pool_gemv
        cpu_raw_simd_gemv.fp8_raw_simd_gemv = cpu_raw_simd_pool_gemv.fp8_raw_simd_pool_gemv
    if os.environ.get("RELIC_HOST_ENGRAM_PREFETCH") == "1":
        from relicllm.models.deepseek_v4_1.host_engram_prefetch import HostEngramPrefetch
        prefetch = HostEngramPrefetch(model).install()
    front.reset_state(1)
    model.temperature = 0.0
    start = time.perf_counter()
    _, logits, _ = front(torch.tensor([ids]), 0)
    sync()
    prefill_seconds = time.perf_counter()-start
    first = int(logits[0].argmax().item())
    position = len(ids)
    exact_boundary = snapshot(model)
    host_boundary = front.snapshot_prefix(model.max_seq_len)

    def reset():
        restore(model, exact_boundary)
        front.restore_prefix(host_boundary)
        sync()

    reference_logits, reference_tokens = [], []
    token = first
    for step in range(QUALITY_STEPS):
        _, logits, _ = front(torch.tensor([[token]]), position+step)
        sync()
        current = logits[0].detach().float().cpu().contiguous()
        if not bool(torch.isfinite(current).all()):
            raise RuntimeError("exact quality control contains nonfinite logits")
        token = int(current.argmax())
        reference_tokens.append(token)
        reference_logits.append(current)
    reset()

    original_linear = ascend_linear.quantized_linear_ascend
    original_hc = modules.hc_split_sinkhorn
    original_projections = [layer.ffn.routed.packed_gemv for layer in model.layers]
    # Validate loaded dense operands before any experimental narrowing. One
    # projection at a time; these load-time checks are outside every TPS window.
    guarded = []
    checked_ids = set()
    from relicllm.models.deepseek_v4_1 import mma
    for name, parameter in model.named_parameters():
        if parameter.device.type == "npu" and parameter.ndim == 2 and mma.attached_codes(parameter) is not None and id(parameter) not in checked_ids:
            checked_ids.add(id(parameter))
            codes, scales = mma.attached_codes(parameter), mma.attached_scale(parameter)
            chunk_reports = []
            for start_row in range(0, parameter.shape[0], 256):
                stop_row = min(start_row+256, parameter.shape[0])
                block_rows = 1 if scales.shape[0] == parameter.shape[0] else 32
                chunk_codes = codes[start_row:stop_row].view(torch.uint8).cpu().contiguous()
                chunk_scales = scales[start_row//block_rows:(stop_row+block_rows-1)//block_rows].cpu().contiguous()
                chunk_reports.append(cube_experiment.check_bf16_representable(chunk_codes, chunk_scales, kind="fp8", name=f"{name}/{start_row}:{stop_row}"))
                chunk_parameter = parameter[start_row:stop_row].cpu().contiguous()
                mma.attach_source_codes(chunk_parameter, chunk_codes.view(torch.float8_e4m3fn), chunk_scales)
                cube_experiment.verify_weight_decode_identity(chunk_parameter)
            guarded.append({"name": name, "row_chunks": len(chunk_reports),
                            "elements": sum(row["elements"] for row in chunk_reports),
                            "non_finite": sum(row["non_finite"] for row in chunk_reports),
                            "not_bf16_representable": sum(row["not_bf16_representable"] for row in chunk_reports)})
    # Every legal E2M1 code at these conservative E8M0 bounds is an exact
    # normal BF16 value (or signed zero). Prove the range, not just its endpoints.
    pattern = torch.tensor([0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]*2, dtype=torch.uint8)
    all_codes = pattern.unsqueeze(0).expand(251,-1).contiguous()
    all_scales = torch.arange(2,253,dtype=torch.int64).to(torch.uint8).unsqueeze(1)
    if not native.scaled_bf16_weight_representable(all_codes, all_scales, fp4=True):
        raise RuntimeError("the conservative E2M1/E8M0 operand range is not exact normal BF16")
    expert_ranges = []
    for layer in model.layers:
        for which, arena in layer.ffn.routed._arenas.items():
            values = arena.scales.to(torch.int32)
            lower, upper = int(values.amin().item()), int(values.amax().item())
            expert_ranges.append({"layer": layer.layer_id, "projection": which, "E8M0_min": lower, "E8M0_max": upper})
            del values
            if lower < 2 or upper > 252:
                raise RuntimeError("a packed expert grid is outside the conservative all-E2M1-normal-BF16 certificate; no narrowing is licensed")
    if variant == "block32":
        projection = partial(native.native_cube_fp4_gemv, mode="block32")
        projection_contract = native.NUMERICAL_CONTRACT
    elif variant == "scaled_bf16":
        projection = partial(native.native_cube_fp4_gemv, mode="scaled_bf16", block_n=32, block_k=128, scaled_bf16_safe=True)
        projection_contract = native.SCALED_BF16_CONTRACT
    elif variant == "lookup":
        from relic_core.kernels.lookup_cube_gemv import fp4_lookup_cube_gemv, CONTRACT
        projection = fp4_lookup_cube_gemv
        projection_contract = CONTRACT
    elif int(os.environ.get("RELIC_CUBE_WEIGHT_CACHE_BYTES", "0")) > 0:
        from relicllm.models.deepseek_v4_1.cube_weight_cache import CubeWeightCache
        budget = int(os.environ["RELIC_CUBE_WEIGHT_CACHE_BYTES"])
        free, total = torch.npu.mem_get_info(next(model.parameters()).device)
        if free < budget + 8*2**30:
            raise RuntimeError("not enough actual HBM headroom for the explicit bounded I8 operand cache")
        projection = CubeWeightCache(budget_bytes=budget,certified_immutable_sources=True)
        projection_contract = "native SDK original I8x2 and half-scales with bounded exact operand cache"
    else:
        projection = cube_experiment.CubePackedProjection(mode=variant, cache_budget_bytes=0, backend="triton")
        projection_contract = "native SDK full-K BF16 accumulation after original scales, no input requantization"
    dense_calls = {"BF16_Cube": 0, "non_BF16_original": 0}
    experimental_linear = cube_experiment.make_linear_provider(original_linear)
    def dense_provider(x, weight, **options):
        # The SDK BF16 matmul has a BF16 final store. Preserve the original
        # path for wider inputs rather than insert an extra output narrowing.
        if x.dtype != torch.bfloat16:
            dense_calls["non_BF16_original"] += 1
            return original_linear(x, weight, **options)
        dense_calls["BF16_Cube"] += 1
        return experimental_linear(x, weight, **options)
    ascend_linear.quantized_linear_ascend = dense_provider
    if os.environ.get("RELIC_HC_FUSED_NORMALIZE") == "1":
        from relic_core.kernels.hc_sinkhorn_normalize import hc_split_sinkhorn_normalized
        modules.hc_split_sinkhorn = hc_split_sinkhorn_normalized
    # `RELIC_EXPERT_PROJECTION=loader` keeps whatever operator the expert store itself installed.
    # It exists so the exact tile-major residency (DEEPSEEK_V41_ASCEND_TILE_MAJOR=1) can be measured
    # against the native-Cube expert operator without also changing the dense path: overriding
    # `packed_gemv` here replaces the store's own dispatch wholesale, so the residency order would
    # otherwise never be exercised.
    keep_loader_expert_projection = os.environ.get("RELIC_EXPERT_PROJECTION", "cube") == "loader"
    if not keep_loader_expert_projection:
        for layer in model.layers:
            layer.ffn.routed.packed_gemv = projection
    expert_projection_source = "loader" if keep_loader_expert_projection else "native_Cube" 
    if os.environ.get("RELIC_FUSED_ORIGINAL_ACTIVATION") == "1":
        from relicllm.models.deepseek_v4_1.fused_activation_binding import FusedActivationBinding
        fused_operator = None
        if os.environ.get("RELIC_FAST_FUSED_ACTIVATION") == "1":
            from relic_core.kernels.fast_fused_fp8_activation_decode import quantize_decode_fp8_fast
            fused_operator = quantize_decode_fp8_fast
        activation_binding = FusedActivationBinding(operator=fused_operator).install()
    expert_tp_binding = None
    if os.environ.get("RELIC_EXPERT_TP", "1") != "1":
        from relic_core.kernels.ascendc_online_tile_cube import AscendCTileCubeConsumer
        from relicllm.models.deepseek_v4_1.ascend_global_ffn import FullFFNGlobalDispatch as ExpertTPDispatch
        tp_consumer = AscendCTileCubeConsumer(os.environ["RELIC_TILE_CONSUMER_LIBRARY"],
            sha256=os.environ["RELIC_TILE_CONSUMER_SHA256"], block_dim=24)
        expert_tp_binding = ExpertTPDispatch(model, consumer=tp_consumer,
            tp_world=int(os.environ["RELIC_EXPERT_TP"]), restore_residency=False).install()
        projection_contract = expert_tp_binding.describe()["numerical_contract"]
        expert_projection_source = "original_expert_tensor_parallel_tile_consumer"
    m1_binding = None
    device_binding = None
    device_tile_moe = os.environ.get("RELIC_DEVICE_TILE_MOE") == "1"
    device_moe = os.environ.get("RELIC_DEVICE_MOE") == "1" or device_tile_moe
    if device_moe:
        if (keep_loader_expert_projection and not device_tile_moe) or os.environ.get("RELIC_EXPERT_BODY_GRAPHS") == "1":
            raise ValueError("device-selected MoE replaces host dispatch; do not install host expert-body graphs")
        if device_tile_moe:
            from relic_core.kernels.ascendc_online_tile_cube import AscendCTileCubeConsumer
            from relicllm.models.deepseek_v4_1.ascend_tile_routing import TileRoutedDispatch
            consumer = AscendCTileCubeConsumer(os.environ["RELIC_TILE_CONSUMER_LIBRARY"],
                                               sha256=os.environ["RELIC_TILE_CONSUMER_SHA256"], block_dim=24)
            device_binding = TileRoutedDispatch(
                [layer.ffn.routed for layer in model.layers], consumer=consumer,
                certified_bf16_operands=True, backend="triton", capture_graphs=True,
                max_workspace_bytes=512 * 2**20).install()
        else:
            from relicllm.models.deepseek_v4_1.ascend_device_routing import DeviceRoutedDispatch
            device_binding = DeviceRoutedDispatch(
                [layer.ffn.routed for layer in model.layers], certified_bf16_operands=True,
                backend="triton", max_workspace_bytes=512 * 2**20, capture_graphs=True,
                projection_mode=os.environ.get("RELIC_DEVICE_MOE_PROJECTION", "grouped"),
                fused_epilogue=os.environ.get("RELIC_DEVICE_MOE_EPILOGUE") == "1",
            ).install()
        projection_contract = device_binding.describe()["numerical_contract"]
        expert_projection_source = "device_selected_native_tile_ring_Cube" if device_tile_moe else "device_selected_native_grouped_Cube"
    elif os.environ.get("RELIC_M1_DIRECT_ROUTING") == "1":
        from relicllm.models.deepseek_v4_1.ascend_m1_routing import M1RoutedDispatch
        m1_binding = M1RoutedDispatch([layer.ffn.routed for layer in model.layers]).install()
    expert_graph_driver = None
    fused_expert_binding = None
    fused_expert_bit_differences = None
    expert_graph_capture = None
    expert_graph_bit_differences = None
    graph_driver = None
    graph_pos = None
    graph_capture_seconds = None
    graph_candidate_bit_differences = None
    def candidate_forward(tokens, at):
        if graph_driver is None:
            return front(tokens, at)
        graph_pos.set(at)
        return front(tokens, graph_pos)
    try:
        teacher_logits, metrics = [], []
        token = first
        for step in range(QUALITY_STEPS):
            _, logits, _ = front(torch.tensor([[token]]), position+step)
            sync()
            current = logits[0].detach().float().cpu().contiguous()
            teacher_logits.append(current)
            metrics.append(compare_logits(current, reference_logits[step], reference_tokens[step]))
            # The exact control's token is fed, not the candidate argmax. Thus
            # logits are compared at the same token context even after a flip.
            token = reference_tokens[step]
        reset()
        if os.environ.get("RELIC_FUSED_EXPERT_EPILOGUE") == "1":
            if device_moe:
                raise ValueError("validate the fused epilogue on the current expert body before combining dispatch experiments")
            from relicllm.models.deepseek_v4_1.fused_expert_epilogue import FusedExpertEpilogue
            fused_expert_binding = FusedExpertEpilogue().install()
            fused_expert_bit_differences = []
            token = first
            for step in range(QUALITY_STEPS):
                _, fused_logits, _ = front(torch.tensor([[token]]), position + step)
                sync()
                current = fused_logits[0].detach().float().cpu().contiguous()
                count = int((current.view(torch.int32) != teacher_logits[step].view(torch.int32)).sum())
                fused_expert_bit_differences.append(count)
                if count:
                    raise RuntimeError("fused expert epilogue changed current-Cube teacher-forced FP32 logit bits")
                token = reference_tokens[step]
            reset()
        if os.environ.get("RELIC_EXPERT_BODY_GRAPHS") == "1":
            from relicllm.models.deepseek_v4_1.ascend_expert_graphs import RoutedExpertGraphs
            expert_graph_driver = RoutedExpertGraphs(model, warmup=1, max_graphs=2048, max_pool_bytes=512*2**20).install()
            expert_graph_capture = expert_graph_driver.capture_all()
            if expert_graph_driver.graphs != 1920:
                raise RuntimeError("the full model did not capture all 1920 owned expert bodies per rank")
            sync()
            reset()
            expert_graph_bit_differences = []
            token = first
            for step in range(QUALITY_STEPS):
                _, logits, _ = front(torch.tensor([[token]]), position+step)
                sync()
                current = logits[0].detach().float().cpu().contiguous()
                differences = int((current.view(torch.int32)!=teacher_logits[step].view(torch.int32)).sum())
                expert_graph_bit_differences.append(differences)
                if differences or not bool(torch.isfinite(current).all()):
                    raise RuntimeError("original expert body graphs changed same-mode eager logit bits")
                token = reference_tokens[step]
            if expert_graph_driver.passthroughs:
                raise RuntimeError("a covered full-model expert did not use its recorded body")
            reset()
        if os.environ.get("RELIC_CUBE_NPUGRAPH") == "1":
            from relicllm.models.deepseek_v4_1.ascend_graphs import NpuDecodeGraphs
            from relicllm.models.deepseek_v4_1.decode_pos import Pos
            graph_driver = NpuDecodeGraphs(model)
            graph_pos = Pos.device(position,next(model.parameters()).device)
            capture_start = time.perf_counter()
            graph_driver.capture_pass(lambda: candidate_forward(torch.tensor([[first]]),position))
            sync()
            graph_capture_seconds = time.perf_counter()-capture_start
            reset()
            graph_candidate_bit_differences = []
            token = first
            for step in range(QUALITY_STEPS):
                _, logits, _ = candidate_forward(torch.tensor([[token]]),position+step)
                sync()
                current = logits[0].detach().float().cpu().contiguous()
                differences = int((current.view(torch.int32)!=teacher_logits[step].view(torch.int32)).sum())
                graph_candidate_bit_differences.append(differences)
                if differences or not bool(torch.isfinite(current).all()):
                    raise RuntimeError("private Cube graph changed same-mode candidate eager logit bits")
                token = reference_tokens[step]
            reset()
        token = first
        for step in range(4):
            _, logits, _ = candidate_forward(torch.tensor([[token]]), position+step)
            sync()
            token = int(logits[0].argmax().item())
        position += 4
        profile_records = None
        if os.environ.get("RELIC_CUBE_PROFILE") == "1":
            from profile_actual_decode import Timers
            from relic_core.kernels import unpack,block_gemm,block_fp8_quant
            from relicllm.models.deepseek_v4_1 import ascend_packed_experts,modules
            import torch_npu
            timed_cache, timed_hashes, timed_token = snapshot(model), front.snapshot_prefix(model.max_seq_len), token
            timer = Timers()
            # Only main-thread paths are timed. CPU prefetch worker bodies are
            # deliberately excluded from this synchronized nested timer stack.
            for owner, method, label in ((ascend_linear,"quantized_linear_ascend","dense.dispatch"),
                    (ascend_packed_experts,"_call_packed_gemv","expert.projection"),
                    (unpack,"unpack_fp4_to_int8","expert.decode"),(unpack,"unpack_fp8_to_bf16","activation.decode"),
                    (unpack,"fp4_int8_scale","expert.half_scale"),(block_gemm,"fp8_block_gemm","dense.exact_remaining"),
                    (block_fp8_quant,"quantize_block_fp8","activation.quantize"),
                    (torch_npu,"npu_weight_quant_batchmatmul","expert.native_Cube"),
                    (modules.Block,"hc_mixes","HC.mixes"),(modules.Block,"hc_post","HC.post"),
                    (modules.Engram,"forward","host.Engram_boundary"),
                    (dist,"all_reduce","collective.all_reduce")):
                timer.patch(owner,method,label)
            try:
                for step in range(2):
                    _, logits, _ = candidate_forward(torch.tensor([[token]]), position+step)
                    sync()
                    token = int(logits[0].argmax().item())
            finally:
                timer.restore()
            profile_records = dict(timer.records)
            restore(model,timed_cache)
            front.restore_prefix(timed_hashes)
            token = timed_token
            sync()
        # Copying quality logits is complete. This contiguous fixed-length
        # window now includes NPU completion and host greedy selection only.
        dist.barrier()
        sync()
        timed_tokens, timings = [], []
        cache_before_timed = projection.describe() if hasattr(projection,"describe") else None
        start_wall = time.perf_counter()
        if TIMED_STEPS != len(MATCHED_INPUTS) or token != MATCHED_INPUTS[0]:
            raise RuntimeError("matched input window has an inconsistent initial prefix")
        for step in range(TIMED_STEPS):
            token = MATCHED_INPUTS[step]
            step_start = time.perf_counter()
            _, logits, _ = candidate_forward(torch.tensor([[token]]), position+step)
            sync()
            token = int(logits[0].argmax().item())
            timings.append(time.perf_counter()-step_start)
            timed_tokens.append(token)
        wall = time.perf_counter()-start_wall
        final = logits[0].detach().float().cpu().contiguous()
        if not bool(torch.isfinite(final).all()):
            raise RuntimeError("timed Cube final logits are nonfinite")
        # Private timing ablations. Each one deliberately breaks the numbers to
        # attribute the remaining per-token cost; none of them is a candidate.
        ablations = None
        if os.environ.get("RELIC_CUBE_ABLATE") == "1":
            ablations = run_ablations(model, candidate_forward, position, first, reset)
        kernel_profile = None
        if os.environ.get("RELIC_KERNEL_PROFILE") == "1":
            kernel_profile = run_kernel_profile(model, candidate_forward, position, first, OUTPUT, reset)
        dispatch_timing = None
        if os.environ.get("RELIC_DISPATCH_TIMING") == "1":
            dispatch_timing = run_dispatch_timing(model, candidate_forward, position, first, reset)
        replay_probe = None
        if os.environ.get("RELIC_REPLAY_PROBE") == "1":
            replay_probe = run_replay_probe(model, expert_graph_driver)
        row = {"rank": rank, "variant": variant, "dense_contract": cube_experiment.MODE,
               "expert_contract": projection_contract, "explicit_numerical_test_permission": True,
               "checkpoint_original_and_not_rewritten": True, "Engram_complete_CPU": True,
               "exact_CPU_reference_reduction_flag": os.environ.get("POCKETLLM_V41_MMA_EMULATE"),
               "quality_context_start": len(ids), "quality_steps": QUALITY_STEPS,
               "teacher_forced_reference_tokens": reference_tokens, "quality_metrics": metrics,
               "prefill_seconds_excluded": prefill_seconds, "prefill_reduction": "original exact",
               "timed_warmup_steps_excluded": 4, "timed_steps": TIMED_STEPS,
               "timed_context_start": position, "timed_context_end": position+TIMED_STEPS,
               "timed_token_ids": timed_tokens, "timed_step_seconds": timings,
               "timed_decode_seconds": wall, "sustained_single_stream_TPS": TIMED_STEPS/wall,
               "single_stream": True, "batch_size": 1, "graphs": graph_driver is not None,
               "private_graph_capture_seconds_excluded": graph_capture_seconds,
               "graph_candidate_eager_FP32_bit_differences": graph_candidate_bit_differences,
               "NPU_completion_and_host_argmax_included": True, "per_step_quality_copies_excluded": True,
               "finite_final_logits": True, "dense_weight_guards": guarded,
               "expert_E8M0_range_guards": expert_ranges, "dense_dispatch_calls": dense_calls,
               "non_BF16_dense_output_precision_preserved_on_original_path": True,
               "fixed_length_may_continue_after_EOS": True,
               "observed_EOS_in_timed_tokens": [index for index, value in enumerate(timed_tokens) if value == kwargs.get("eos_token_id")],
               "packed_residency": [layer.ffn.routed.cache_stats for layer in model.layers],
               "host_engram_states": [layer.engram._host_engram_state.describe() for layer in model.layers if layer.engram is not None],
               "production_defaults_and_serving_guards_unchanged": True,
               "exact_CPU_worker_pool": os.environ.get("RELIC_EXACT_CPU_POOL") == "1",
               "host_Engram_prefetch": None if prefetch is None else prefetch.describe(),
               "two_step_nested_profile_excluded_from_TPS": profile_records,
               "timing_ablations_excluded_from_TPS": ablations,
               "graph_replay_latency_excluded_from_TPS": replay_probe,
               "dispatch_host_timing_excluded_from_TPS": dispatch_timing,
               "kernel_profile_not_a_TPS_window": kernel_profile,
               "expert_projection_source": expert_projection_source,
               "tile_major_expert_residency": os.environ.get("DEEPSEEK_V41_ASCEND_TILE_MAJOR") == "1",
               "exact_I8_operand_cache_before_timed": cache_before_timed,
               "HC_FP32_normalization_native_reduction": os.environ.get("RELIC_HC_FUSED_NORMALIZE") == "1",
               "HC_original_sigmoid_softmax_and_FP32_weight_values_unchanged": True,
               "exact_vector_FP4_widen": os.environ.get("RELIC_VECTOR_FP4_WIDEN") == "1",
               "exact_fused_original_activation_binding": None if activation_binding is None else activation_binding.describe(),
               "M1_direct_routing_binding": None if m1_binding is None else m1_binding.describe(),
               "device_selected_MoE": None if device_binding is None else device_binding.describe(),
               "fused_expert_epilogue": None if fused_expert_binding is None else fused_expert_binding.describe(),
               "fused_expert_vs_current_Cube_FP32_bit_differences": fused_expert_bit_differences,
               "expert_body_graph_capture_excluded_from_TPS": expert_graph_capture,
               "expert_body_graph_teacher_FP32_bit_differences": expert_graph_bit_differences,
               "expert_body_graphs": None if expert_graph_driver is None else expert_graph_driver.describe(),
               "exact_I8_operand_cache": projection.describe() if hasattr(projection,"describe") else None}
        row["same_token_input_latency_not_generation_TPS"] = True
        row["fixed_input_token_ids"] = MATCHED_INPUTS
        row["single_request_forward_rate_not_generation_TPS"] = row.pop("sustained_single_stream_TPS")
        row["NPU_completion_and_host_argmax_included"] = True
        row["expert_TP_EP"] = None if expert_tp_binding is None else expert_tp_binding.describe()
        with (OUTPUT/f"Cube-model-rank-{rank}.json").open("x") as handle:
            json.dump(row, handle, indent=2)
        torch.save({"reference": torch.stack(reference_logits), "candidate_teacher_forced": torch.stack(teacher_logits)}, OUTPUT/f"Cube-quality-logits-rank-{rank}.pt")
        torch.save(final, OUTPUT/f"Cube-final-logits-rank-{rank}.pt")
        print("ISOLATED_CUBE_MODEL_RESULT", json.dumps({key: row[key] for key in ("rank", "variant", "timed_steps", "timed_decode_seconds", "single_request_forward_rate_not_generation_TPS", "timed_token_ids")}), flush=True)
    finally:
        if graph_driver is not None:
            graph_driver.release()
        if expert_graph_driver is not None:
            expert_graph_driver.release()
        if fused_expert_binding is not None:
            fused_expert_binding.close()
        if activation_binding is not None:
            activation_binding.close()
        if m1_binding is not None:
            m1_binding.close()
        if device_binding is not None:
            device_binding.close()
        if expert_tp_binding is not None:
            expert_tp_binding.close(restore_residency=False)
        ascend_linear.quantized_linear_ascend = original_linear
        modules.hc_split_sinkhorn = original_hc
        for layer, original in zip(model.layers, original_projections):
            layer.ffn.routed.packed_gemv = original
        if hasattr(projection, "clear"):
            projection.clear()
        if prefetch is not None:
            prefetch.close(propagate_errors=False)
    result = Generation(prompt_tokens=len(ids))
    result.tokens = timed_tokens
    result.decode_seconds = wall
    result.stopped = "isolated_authorized_Cube_probe"
    return result
