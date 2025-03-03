// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "decoder_write_cache_with_rope_kernel.h"
#include "utils.cuh"


template <typename T>
void DecoderWriteCacheKV(const AppendAttnMetaData& meta_data,
                         const paddle::Tensor& qkv,
                         const paddle::Tensor& seq_lens,
                         const paddle::Tensor& seq_lens_encoder,
                         const paddle::Tensor& padding_offsets,
                         const paddle::Tensor& cum_offsets,
                         const paddle::Tensor& block_tables,
                         const int max_seq_len,
                         cudaStream_t& stream,
                         paddle::Tensor* key_cache_out,
                         paddle::Tensor* value_cache_out) {
  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;
  auto bsz = meta_data.batch_size;
  auto block_size = meta_data.block_size;
  auto head_dim_qk = meta_data.head_dims;
  auto head_dim_v = meta_data.head_dims_v;
  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  const uint32_t elem_nums = bsz * kv_num_heads * (head_dim_qk + head_dim_v);

  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);

  append_decode_cache_T_kernel<T, PackSize>
      <<<grid_size, blocksize, 0, stream>>>(
          reinterpret_cast<T*>(const_cast<T*>(qkv.data<T>())),
          reinterpret_cast<T*>(key_cache_out->data<T>()),
          reinterpret_cast<T*>(value_cache_out->data<T>()),
          block_tables.data<int>(),
          padding_offsets.data<int>(),
          cum_offsets.data<int>(),
          seq_lens.data<int>(),
          seq_lens_encoder.data<int>(),
          max_seq_len,
          max_blocks_per_seq,
          num_heads,
          head_dim_qk,
          head_dim_v,
          block_size,
          elem_nums,
          kv_num_heads);
}

template <typename T, typename QKV_TYPE>
void append_decode_cache_rope(const QKV_TYPE* qkv,
                              T* key_cache,
                              T* value_cache,
                              T* qkv_out,
                              const int* block_tables,
                              const int* padding_offsets,
                              const int* cum_offsets,
                              const int* seq_lens,
                              const int* seq_lens_encoder,
                              const float* cos_emb,
                              const float* sin_emb,
                              const float* qkv_out_scales,
                              const T* qkv_biases,
                              const int max_seq_len,
                              const int max_blocks_per_seq,
                              const int num_heads,
                              const int kv_num_heads,
                              const int dim_head,
                              const int block_size,
                              const int bsz,
                              const cudaStream_t& stream,
                              const bool use_neox_style) {
  const uint32_t elem_nums =
      use_neox_style ? bsz * (num_heads + 2 * kv_num_heads) * dim_head / 2
                     : bsz * (num_heads + 2 * kv_num_heads) * dim_head;

  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);
  if (use_neox_style) {
    if (qkv_out_scales) {
      append_decode_cache_T_neox_rope_kernel<T, PackSize>
          <<<grid_size, blocksize, 0, stream>>>(
              reinterpret_cast<const int*>(qkv),
              key_cache,
              value_cache,
              qkv_out,
              block_tables,
              padding_offsets,
              cum_offsets,
              seq_lens,
              seq_lens_encoder,
              cos_emb,
              sin_emb,
              qkv_out_scales,
              qkv_biases,
              max_seq_len,
              max_blocks_per_seq,
              num_heads,
              dim_head,
              block_size,
              elem_nums,
              kv_num_heads);
    } else {
      append_decode_cache_T_neox_rope_kernel<T, PackSize>
          <<<grid_size, blocksize, 0, stream>>>(reinterpret_cast<const T*>(qkv),
                                                key_cache,
                                                value_cache,
                                                qkv_out,
                                                block_tables,
                                                padding_offsets,
                                                cum_offsets,
                                                seq_lens,
                                                seq_lens_encoder,
                                                cos_emb,
                                                sin_emb,
                                                max_seq_len,
                                                max_blocks_per_seq,
                                                num_heads,
                                                dim_head,
                                                block_size,
                                                elem_nums,
                                                kv_num_heads);
    }
  } else {
    if (qkv_out_scales) {
      append_decode_cache_T_rope_kernel<T, PackSize>
          <<<grid_size, blocksize, 0, stream>>>(
              reinterpret_cast<const int*>(qkv),
              key_cache,
              value_cache,
              qkv_out,
              block_tables,
              padding_offsets,
              cum_offsets,
              seq_lens,
              seq_lens_encoder,
              cos_emb,
              sin_emb,
              qkv_out_scales,
              qkv_biases,
              max_seq_len,
              max_blocks_per_seq,
              num_heads,
              dim_head,
              block_size,
              elem_nums,
              kv_num_heads);
    } else {
      append_decode_cache_T_rope_kernel<T, PackSize>
          <<<grid_size, blocksize, 0, stream>>>(reinterpret_cast<const T*>(qkv),
                                                key_cache,
                                                value_cache,
                                                qkv_out,
                                                block_tables,
                                                padding_offsets,
                                                cum_offsets,
                                                seq_lens,
                                                seq_lens_encoder,
                                                cos_emb,
                                                sin_emb,
                                                max_seq_len,
                                                max_blocks_per_seq,
                                                num_heads,
                                                dim_head,
                                                block_size,
                                                elem_nums,
                                                kv_num_heads);
    }
  }
}

template <typename T, typename QKV_TYPE>
void append_decode_cache_int8_rope(const QKV_TYPE* qkv,
                                   uint8_t* key_cache,
                                   uint8_t* value_cache,
                                   T* qkv_out,
                                   const int* block_tables,
                                   const int* padding_offsets,
                                   const int* cum_offsets,
                                   const int* seq_lens,
                                   const int* seq_lens_encoder,
                                   const float* cos_emb,
                                   const float* sin_emb,
                                   const float* qkv_out_scales,
                                   const T* qkv_biases,
                                   const T* cache_k_scale,
                                   const T* cache_v_scale,
                                   const int max_seq_len,
                                   const int max_blocks_per_seq,
                                   const int num_heads,
                                   const int kv_num_heads,
                                   const int dim_head,
                                   const int block_size,
                                   const int bsz,
                                   const cudaStream_t& stream,
                                   const bool use_neox_style) {
  constexpr int num_warps = 4;
  const int all_warps =
      ((num_heads + 2 * kv_num_heads) + num_warps - 1) / num_warps * num_warps;
  dim3 grids(bsz, all_warps / num_warps);
  if (use_neox_style) {
    if (qkv_out_scales) {
      append_decode_cache_int8_neox_rope_kernel<T, 4>
          <<<grids, num_warps * 32, 0, stream>>>(
              reinterpret_cast<const int*>(qkv),
              key_cache,
              value_cache,
              qkv_out,
              block_tables,
              padding_offsets,
              cum_offsets,
              seq_lens,
              seq_lens_encoder,
              cos_emb,
              sin_emb,
              qkv_out_scales,
              qkv_biases,
              cache_k_scale,
              cache_v_scale,
              max_seq_len,
              max_blocks_per_seq,
              num_heads,
              block_size,
              127.0f,
              -127.0f,
              kv_num_heads);
    } else {
      append_decode_cache_int8_neox_rope_kernel<T, 4>
          <<<grids, num_warps * 32, 0, stream>>>(
              reinterpret_cast<const T*>(qkv),
              key_cache,
              value_cache,
              qkv_out,
              block_tables,
              padding_offsets,
              cum_offsets,
              seq_lens,
              seq_lens_encoder,
              cos_emb,
              sin_emb,
              cache_k_scale,
              cache_v_scale,
              max_seq_len,
              max_blocks_per_seq,
              num_heads,
              block_size,
              127.0f,
              -127.0f,
              kv_num_heads);
    }
  } else {
    if (qkv_out_scales) {
      append_decode_cache_int8_rope_kernel<T, 4>
          <<<grids, num_warps * 32, 0, stream>>>(
              reinterpret_cast<const int*>(qkv),
              key_cache,
              value_cache,
              qkv_out,
              block_tables,
              padding_offsets,
              cum_offsets,
              seq_lens,
              seq_lens_encoder,
              cos_emb,
              sin_emb,
              qkv_out_scales,
              qkv_biases,
              cache_k_scale,
              cache_v_scale,
              max_seq_len,
              max_blocks_per_seq,
              num_heads,
              block_size,
              127.0f,
              -127.0f,
              kv_num_heads);
    } else {
      append_decode_cache_int8_rope_kernel<T, 4>
          <<<grids, num_warps * 32, 0, stream>>>(
              reinterpret_cast<const T*>(qkv),
              key_cache,
              value_cache,
              qkv_out,
              block_tables,
              padding_offsets,
              cum_offsets,
              seq_lens,
              seq_lens_encoder,
              cos_emb,
              sin_emb,
              cache_k_scale,
              cache_v_scale,
              max_seq_len,
              max_blocks_per_seq,
              num_heads,
              block_size,
              127.0f,
              -127.0f,
              kv_num_heads);
    }
  }
}

template <typename T, typename QKV_TYPE>
void append_decode_cache_int4_rope(const QKV_TYPE* qkv,
                                   uint8_t* key_cache,
                                   uint8_t* value_cache,
                                   T* qkv_out,
                                   const int* block_tables,
                                   const int* padding_offsets,
                                   const int* cum_offsets,
                                   const int* seq_lens,
                                   const int* seq_lens_encoder,
                                   const float* cos_emb,
                                   const float* sin_emb,
                                   const float* qkv_out_scales,
                                   const T* qkv_biases,
                                   const T* cache_k_scale,
                                   const T* cache_v_scale,
                                   const T* cache_k_zp,
                                   const T* cache_v_zp,
                                   const int max_seq_len,
                                   const int max_blocks_per_seq,
                                   const int num_heads,
                                   const int kv_num_heads,
                                   const int dim_head,
                                   const int block_size,
                                   const int bsz,
                                   const cudaStream_t& stream,
                                   const bool use_neox_style) {
  constexpr int num_warps = 4;
  const int all_warps =
      ((num_heads + 2 * kv_num_heads) + num_warps - 1) / num_warps * num_warps;
  dim3 grids(bsz, all_warps / num_warps);
  if (use_neox_style) {
    if (qkv_out_scales) {
      append_decode_cache_int4_neox_rope_kernel<T, 4>
          <<<grids, num_warps * 32, 0, stream>>>(
              reinterpret_cast<const int*>(qkv),
              key_cache,
              value_cache,
              qkv_out,
              block_tables,
              padding_offsets,
              cum_offsets,
              seq_lens,
              seq_lens_encoder,
              cos_emb,
              sin_emb,
              qkv_out_scales,
              qkv_biases,
              cache_k_scale,
              cache_v_scale,
              cache_k_zp,
              cache_v_zp,
              max_seq_len,
              max_blocks_per_seq,
              num_heads,
              block_size,
              7.0f,
              -8.0f,
              kv_num_heads);
    } else {
      append_decode_cache_int4_neox_rope_kernel<T, 4>
          <<<grids, num_warps * 32, 0, stream>>>(
              reinterpret_cast<const T*>(qkv),
              key_cache,
              value_cache,
              qkv_out,
              block_tables,
              padding_offsets,
              cum_offsets,
              seq_lens,
              seq_lens_encoder,
              cos_emb,
              sin_emb,
              cache_k_scale,
              cache_v_scale,
              cache_k_zp,
              cache_v_zp,
              max_seq_len,
              max_blocks_per_seq,
              num_heads,
              block_size,
              7.0f,
              -8.0f,
              kv_num_heads);
    }
  } else {
    if (qkv_out_scales) {
      append_decode_cache_int4_rope_kernel<T, 4>
          <<<grids, num_warps * 32, 0, stream>>>(
              reinterpret_cast<const int*>(qkv),
              key_cache,
              value_cache,
              qkv_out,
              block_tables,
              padding_offsets,
              cum_offsets,
              seq_lens,
              seq_lens_encoder,
              cos_emb,
              sin_emb,
              qkv_out_scales,
              qkv_biases,
              cache_k_scale,
              cache_v_scale,
              cache_k_zp,
              cache_v_zp,
              max_seq_len,
              max_blocks_per_seq,
              num_heads,
              block_size,
              7.0f,
              -8.0f,
              kv_num_heads);
    } else {
      append_decode_cache_int4_rope_kernel<T, 4>
          <<<grids, num_warps * 32, 0, stream>>>(
              reinterpret_cast<const T*>(qkv),
              key_cache,
              value_cache,
              qkv_out,
              block_tables,
              padding_offsets,
              cum_offsets,
              seq_lens,
              seq_lens_encoder,
              cos_emb,
              sin_emb,
              cache_k_scale,
              cache_v_scale,
              cache_k_zp,
              cache_v_zp,
              max_seq_len,
              max_blocks_per_seq,
              num_heads,
              block_size,
              7.0f,
              -8.0f,
              kv_num_heads);
    }
  }
}
template <typename T, typename QKV_TYPE>
void DecoderWriteCacheWithRoPEKernel(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& qkv,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_seq_len,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out) {
  typedef cascade_attn_type_traits<T> traits_;
  typedef cascade_attn_type_traits<QKV_TYPE> qkt_nv_type_;
  typedef typename traits_::type DataType_;
  typedef typename qkt_nv_type_::type QKV_Data_TYPE;
  const QKV_TYPE* qkv_ptr = qkv.data<QKV_TYPE>();

  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;
  auto bsz = meta_data.batch_size;
  auto block_size = meta_data.block_size;
  auto dim_head = meta_data.head_dims;
  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;

  if (rotary_embs) {
    const float* cos_emb = rotary_embs.get().data<float>();
    const float* sin_emb =
        use_neox_rotary_style
            ? rotary_embs.get().data<float>() + max_seq_len * dim_head
            : rotary_embs.get().data<float>() + max_seq_len * dim_head / 2;
    if (cache_quant_type_str == "none") {
      append_decode_cache_rope(
          reinterpret_cast<const QKV_TYPE*>(qkv_ptr),
          reinterpret_cast<DataType_*>(key_cache_out->data<T>()),
          reinterpret_cast<DataType_*>(value_cache_out->data<T>()),
          reinterpret_cast<DataType_*>(qkv_out->data<T>()),
          block_tables.data<int>(),
          padding_offsets.data<int>(),
          cum_offsets.data<int>(),
          seq_lens.data<int>(),
          seq_lens_encoder.data<int>(),
          cos_emb,
          sin_emb,
          qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
          qkv_biases ? reinterpret_cast<DataType_*>(
                           const_cast<T*>(qkv_biases.get().data<T>()))
                     : nullptr,
          max_seq_len,
          max_blocks_per_seq,
          num_heads,
          kv_num_heads,
          dim_head,
          block_size,
          bsz,
          stream,
          use_neox_rotary_style);
    } else if (cache_quant_type_str == "cache_int8") {
      append_decode_cache_int8_rope(
          reinterpret_cast<const QKV_TYPE*>(qkv_ptr),
          key_cache_out->data<uint8_t>(),
          value_cache_out->data<uint8_t>(),
          reinterpret_cast<DataType_*>(qkv_out->data<T>()),
          block_tables.data<int>(),
          padding_offsets.data<int>(),
          cum_offsets.data<int>(),
          seq_lens.data<int>(),
          seq_lens_encoder.data<int>(),
          cos_emb,
          sin_emb,
          qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
          qkv_biases ? reinterpret_cast<DataType_*>(
                           const_cast<T*>(qkv_biases.get().data<T>()))
                     : nullptr,
          cache_k_scale ? reinterpret_cast<DataType_*>(
                              const_cast<T*>(cache_k_scale.get().data<T>()))
                        : nullptr,
          cache_v_scale ? reinterpret_cast<DataType_*>(
                              const_cast<T*>(cache_v_scale.get().data<T>()))
                        : nullptr,
          max_seq_len,
          max_blocks_per_seq,
          num_heads,
          kv_num_heads,
          dim_head,
          block_size,
          bsz,
          stream,
          use_neox_rotary_style);
    } else if (cache_quant_type_str == "cache_int4_zp") {
      append_decode_cache_int4_rope(
          reinterpret_cast<const QKV_TYPE*>(qkv_ptr),
          key_cache_out->data<uint8_t>(),
          value_cache_out->data<uint8_t>(),
          reinterpret_cast<DataType_*>(const_cast<T*>(qkv_out->data<T>())),
          block_tables.data<int>(),
          padding_offsets.data<int>(),
          cum_offsets.data<int>(),
          seq_lens.data<int>(),
          seq_lens_encoder.data<int>(),
          cos_emb,
          sin_emb,
          qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
          qkv_biases ? reinterpret_cast<DataType_*>(
                           const_cast<T*>(qkv_biases.get().data<T>()))
                     : nullptr,
          cache_k_scale ? reinterpret_cast<DataType_*>(
                              const_cast<T*>(cache_k_scale.get().data<T>()))
                        : nullptr,
          cache_v_scale ? reinterpret_cast<DataType_*>(
                              const_cast<T*>(cache_v_scale.get().data<T>()))
                        : nullptr,
          cache_k_zp ? reinterpret_cast<DataType_*>(
                           const_cast<T*>(cache_k_zp.get().data<T>()))
                     : nullptr,
          cache_v_zp ? reinterpret_cast<DataType_*>(
                           const_cast<T*>(cache_v_zp.get().data<T>()))
                     : nullptr,
          max_seq_len,
          max_blocks_per_seq,
          num_heads,
          kv_num_heads,
          dim_head,
          block_size,
          bsz,
          stream,
          use_neox_rotary_style);
    } else {
      PD_THROW(
          "cache_quant_type_str should be one of [none, cache_int8, "
          "cache_int4_zp]");
    }
  } else {
    DecoderWriteCacheKV<QKV_TYPE>(meta_data,
                                  qkv,
                                  seq_lens,
                                  seq_lens_encoder,
                                  padding_offsets,
                                  cum_offsets,
                                  block_tables,
                                  max_seq_len,
                                  stream,
                                  key_cache_out,
                                  value_cache_out);
  }
}


template void DecoderWriteCacheWithRoPEKernel<paddle::bfloat16, int>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor&
        qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
              // kv_num_heads, head_dim] if GQA)
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_seq_len,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out);

template void
DecoderWriteCacheWithRoPEKernel<paddle::bfloat16, paddle::bfloat16>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor&
        qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
              // kv_num_heads, head_dim] if GQA)
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_seq_len,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out);

template void DecoderWriteCacheWithRoPEKernel<paddle::float16, int>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor&
        qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
              // kv_num_heads, head_dim] if GQA)
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_seq_len,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out);

template void DecoderWriteCacheWithRoPEKernel<paddle::float16, paddle::float16>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor&
        qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
              // kv_num_heads, head_dim] if GQA)
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_seq_len,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out);