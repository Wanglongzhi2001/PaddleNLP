#include "paddle/extension.h"

__global__ void SpeculateRemovePadding(int64_t *output_data,
                              const int64_t *input_data,
                              const int64_t *draft_tokens,
                              const int *seq_lens,
                              const int *seq_lens_encoder,
                              const int *cum_offsets,
                              const int sequence_length,
                              const int max_draft_tokens) {
  const int bi = blockIdx.x;
  const int tid = threadIdx.x;

  for (int i = tid; i < seq_lens[bi]; i += blockDim.x) {
    const int tgt_seq_id = bi * sequence_length - cum_offsets[bi] + i;
    if (seq_lens_encoder[bi] > 0) {
      const int src_seq_id = bi * sequence_length + i;
      output_data[tgt_seq_id] = input_data[src_seq_id];
    } else {
      const int src_seq_id = bi * max_draft_tokens + i;
      output_data[tgt_seq_id] = draft_tokens[src_seq_id];
    }
  }
}

__global__ void SpeculateGetPaddingOffsetKernel(int *padding_offset,
                                       int *cum_offsets_out,
                                       int *cu_seqlens_q,
                                       int *cu_seqlens_k,
                                       const int *cum_offsets,
                                       const int *seq_lens,
                                       const int max_seq_len) {
  // get padding offset of each batch
  const int bi = blockIdx.x;
  const int ti = threadIdx.x;
  int cum_offset = bi == 0 ? 0 : cum_offsets[bi - 1];
  for (int i = ti; i < seq_lens[bi]; i += blockDim.x) {
    padding_offset[bi * max_seq_len - cum_offset + i] = cum_offset;
  }
  if (ti == 0) {
    cum_offsets_out[bi] = cum_offset;
    int cum_seq_len = (bi + 1) * max_seq_len - cum_offsets[bi];
    cu_seqlens_q[bi + 1] = cum_seq_len;
    cu_seqlens_k[bi + 1] = cum_seq_len;
  }
}


std::vector<paddle::Tensor> SpeculateGetPaddingOffset(const paddle::Tensor& input_ids,
                                             const paddle::Tensor& draft_tokens,
                                             const paddle::Tensor& cum_offsets,
                                             const paddle::Tensor& token_num,
                                             const paddle::Tensor& seq_len,
                                             const paddle::Tensor& seq_lens_encoder) {
    auto cu_stream = input_ids.stream();
    std::vector<int64_t> input_ids_shape = input_ids.shape();
    const int bsz = seq_len.shape()[0];
    const int seq_length = input_ids_shape[1];
    const int max_draft_tokens = draft_tokens.shape()[1];
    auto cum_offsets_out = cum_offsets.copy_to(cum_offsets.place(), false);
    auto cpu_token_num = token_num.copy_to(paddle::CPUPlace(), false);

    const int token_num_data = cpu_token_num.data<int64_t>()[0];
    auto x_remove_padding = paddle::full({token_num_data}, 0, paddle::DataType::INT64, input_ids.place());
    auto padding_offset = paddle::full({token_num_data}, 0, paddle::DataType::INT32, input_ids.place());
    auto cu_seqlens_q = paddle::full({bsz + 1}, 0, paddle::DataType::INT32, input_ids.place());
    auto cu_seqlens_k = paddle::full({bsz + 1}, 0, paddle::DataType::INT32, input_ids.place());
    int blockSize = min((token_num_data + 32 - 1) / 32 * 32, 128);
    SpeculateGetPaddingOffsetKernel<<<bsz, 128, 0, cu_stream>>>(
      padding_offset.data<int>(), 
      cum_offsets_out.data<int>(),
      cu_seqlens_q.data<int>(),
      cu_seqlens_k.data<int>(),
      cum_offsets.data<int>(),
      seq_len.data<int>(),
      seq_length);
    SpeculateRemovePadding<<<bsz, blockSize, 0, cu_stream>>>(
      x_remove_padding.data<int64_t>(), 
      input_ids.data<int64_t>(), 
      draft_tokens.data<int64_t>(),
      seq_len.data<int>(),
      seq_lens_encoder.data<int>(),
      cum_offsets_out.data<int>(), 
      seq_length,
      max_draft_tokens);
    return {x_remove_padding, cum_offsets_out, padding_offset, cu_seqlens_q, cu_seqlens_k}; // , enc_token_num, dec_token_num};
}

std::vector<std::vector<int64_t>> SpeculateGetPaddingOffsetInferShape(const std::vector<int64_t>& input_ids_shape,
                                                             const std::vector<int64_t>& draft_tokens_shape,
                                                             const std::vector<int64_t>& cum_offsets_shape,
                                                             const std::vector<int64_t>& token_num_shape,
                                                             const std::vector<int64_t>& seq_len_shape,
                                                             const std::vector<int64_t>& seq_lens_encoder_shape) {
    int64_t bsz = seq_len_shape[0];
    int64_t seq_len = input_ids_shape[1];
    return {{-1}, {bsz}, {-1}, {bsz + 1}, {bsz + 1}};
}

std::vector<paddle::DataType> SpeculateGetPaddingOffsetInferDtype(const paddle::DataType& input_ids_dtype,
                                                         const paddle::DataType& draft_tokens_dtype,
                                                         const paddle::DataType& cum_offsets_dtype,
                                                         const paddle::DataType& token_num_dtype,
                                                         const paddle::DataType& seq_len_dtype,
                                                         const paddle::DataType& seq_lens_encoder_dtype) {
    return {input_ids_dtype, seq_len_dtype, seq_len_dtype, seq_len_dtype, seq_len_dtype};
}

PD_BUILD_OP(speculate_get_padding_offset)
    .Inputs({"input_ids", "draft_tokens", "token_num", "cum_offsets", "seq_len", "seq_lens_encoder"})
    .Outputs({"x_remove_padding", "cum_offsets_out", "padding_offset", "cu_seqlens_q", "cu_seqlens_k"})
    .SetKernelFn(PD_KERNEL(SpeculateGetPaddingOffset))
    .SetInferShapeFn(PD_INFER_SHAPE(SpeculateGetPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SpeculateGetPaddingOffsetInferDtype));