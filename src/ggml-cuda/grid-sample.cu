#include "grid-sample.cuh"

#define CUDA_GRID_SAMPLE_2D_BLOCK_SIZE 256

static __device__ __forceinline__ float grid_sample_2d_coord(float coord, int64_t size, bool align_corners) {
    if (coord != coord) {
        coord = -1.0f;
    }

    if (align_corners) {
        return 0.5f*(coord + 1.0f)*(float)(size - 1);
    }

    return 0.5f*(coord + 1.0f)*(float)size - 0.5f;
}

static __device__ __forceinline__ float grid_sample_2d_get_f32(
        const float * src,
        int64_t       nb0,
        int64_t       nb1,
        int64_t       nb2,
        int64_t       nb3,
        int64_t       width,
        int64_t       height,
        int64_t       x,
        int64_t       y,
        int64_t       channel,
        int64_t       batch) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0.0f;
    }

    return *(const float *)((const char *) src + x*nb0 + y*nb1 + channel*nb2 + batch*nb3);
}

static __global__ void grid_sample_2d_f32(
        const float * src,
        const float * grid,
        float *       dst,
        int64_t       nb00,
        int64_t       nb01,
        int64_t       nb02,
        int64_t       nb03,
        int64_t       nb10,
        int64_t       nb11,
        int64_t       nb12,
        int64_t       nb13,
        int64_t       nb0,
        int64_t       nb1,
        int64_t       nb2,
        int64_t       nb3,
        int64_t       src_width,
        int64_t       src_height,
        int64_t       dst_width,
        int64_t       dst_height,
        int64_t       channels,
        int64_t       batch_count,
        bool          align_corners) {
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    const int64_t n = dst_width*dst_height*channels*batch_count;

    if (i >= n) {
        return;
    }

    const int64_t x_out   = i % dst_width;
    const int64_t y_out   = (i / dst_width) % dst_height;
    const int64_t channel = (i / (dst_width*dst_height)) % channels;
    const int64_t batch   = i / (dst_width*dst_height*channels);

    const float grid_x = *(const float *)((const char *) grid + 0*nb10 + x_out*nb11 + y_out*nb12 + batch*nb13);
    const float grid_y = *(const float *)((const char *) grid + 1*nb10 + x_out*nb11 + y_out*nb12 + batch*nb13);

    const float x_src = grid_sample_2d_coord(grid_x, src_width,  align_corners);
    const float y_src = grid_sample_2d_coord(grid_y, src_height, align_corners);

    const int64_t x0 = (int64_t) floorf(x_src);
    const int64_t y0 = (int64_t) floorf(y_src);
    const int64_t x1 = x0 + 1;
    const int64_t y1 = y0 + 1;

    const float dx = x_src - (float) x0;
    const float dy = y_src - (float) y0;

    const float v00 = grid_sample_2d_get_f32(src, nb00, nb01, nb02, nb03, src_width, src_height, x0, y0, channel, batch);
    const float v01 = grid_sample_2d_get_f32(src, nb00, nb01, nb02, nb03, src_width, src_height, x0, y1, channel, batch);
    const float v10 = grid_sample_2d_get_f32(src, nb00, nb01, nb02, nb03, src_width, src_height, x1, y0, channel, batch);
    const float v11 = grid_sample_2d_get_f32(src, nb00, nb01, nb02, nb03, src_width, src_height, x1, y1, channel, batch);

    const float value =
        v00*(1.0f - dx)*(1.0f - dy) +
        v10*dx*(1.0f - dy) +
        v01*(1.0f - dx)*dy +
        v11*dx*dy;

    *(float *)((char *) dst + x_out*nb0 + y_out*nb1 + channel*nb2 + batch*nb3) = value;
}

void ggml_cuda_op_grid_sample_2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * grid = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(grid->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(grid->ne[0] == 2);
    GGML_ASSERT(grid->ne[1] == dst->ne[0]);
    GGML_ASSERT(grid->ne[2] == dst->ne[1]);
    GGML_ASSERT(grid->ne[3] == dst->ne[3]);
    GGML_ASSERT(src0->ne[2] == dst->ne[2]);
    GGML_ASSERT(src0->ne[3] == dst->ne[3]);
    GGML_ASSERT((enum ggml_grid_sample_mode) ggml_get_op_params_i32(dst, 0) == GGML_GRID_SAMPLE_MODE_BILINEAR);
    GGML_ASSERT((enum ggml_grid_sample_padding) ggml_get_op_params_i32(dst, 1) == GGML_GRID_SAMPLE_PADDING_ZEROS);

    cudaStream_t stream = ctx.stream();

    const int64_t n = ggml_nelements(dst);
    const int64_t num_blocks = (n + CUDA_GRID_SAMPLE_2D_BLOCK_SIZE - 1) / CUDA_GRID_SAMPLE_2D_BLOCK_SIZE;
    const bool align_corners = ggml_get_op_params_i32(dst, 2) != 0;

    grid_sample_2d_f32<<<num_blocks, CUDA_GRID_SAMPLE_2D_BLOCK_SIZE, 0, stream>>>(
        (const float *) src0->data,
        (const float *) grid->data,
        (float *) dst->data,
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        grid->nb[0], grid->nb[1], grid->nb[2], grid->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        src0->ne[0], src0->ne[1],
        dst->ne[0], dst->ne[1],
        dst->ne[2], dst->ne[3],
        align_corners);
}
