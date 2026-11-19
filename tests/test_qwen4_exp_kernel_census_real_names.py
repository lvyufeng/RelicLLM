"""Real native trace names retain the operation, not a mangling prefix."""
from analyze_qwen4_exp_kernel_census import classify


def test_real_gdn_scan_name_is_not_hidden_as_unclassified():
    assert classify('void (anonymous namespace)::gated_delta_bf16_kernel<1>(float const*, float*)')=='gdn_recurrent'


def test_real_cat_copy_and_depthwise_names_keep_actionable_families():
    assert classify('void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<OpaqueType<2u>>')=='copy'
    assert classify('void at::native::(anonymous namespace)::conv_depthwise2d_forward_kernel_generic<c10::BFloat16,int>')=='convolution'
