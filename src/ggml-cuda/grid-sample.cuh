#pragma once

#include "common.cuh"

void ggml_cuda_op_grid_sample_2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
