# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

import paddle
from paddlenlp_ops import ngram_match


class Proposer(ABC):
    """
    Abstract base class for all proposers that can be used in the speculative decoding framework.
    The subclasses of this class must implement the run method to get the draft tokens that are
    generated by the proposer.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def run(self, model_inputs, **kargs):
        """
        Get the draft tokens that are generated by the proposer.
        """
        raise NotImplementedError()


class InferenceWithReferenceProposer(Proposer):
    """
    InferenceWithReference(https://arxiv.org/pdf/2304.04487) is one of the speculative decoding method.
    It match tokens in the input and output as draft tokens.
    """

    def __init__(self, max_draft_tokens, max_ngram_size, max_batch_size, max_seq_len, **kwargs):
        """
        Args:
        max_draft_tokens (int):
            Maximum number of tokens a proposer can generate at one time.
            The hyperparameter of k in the paper.
        max_ngram_size (int):
            The maximum size of the window used to match inputs and outputs.
            The hyperparameter of n in the paper.
        max_batch_size (int):
            The maximum batch size.
        """
        super().__init__()
        self.max_ngram_size = max_ngram_size
        self.input_ids_len = paddle.zeros(shape=[max_batch_size, 1], dtype="int64").cpu()
        self.max_batch_size = max_batch_size
        self.max_draft_tokens = max_draft_tokens
        self.input_ids_cpu = paddle.full(shape=[max_batch_size, max_seq_len], fill_value=1, dtype="int64").cpu()

    # TODO(Wanglongzhi2001): unnessary
    # def update(self, bid: int, seq_len: int):
    #     """
    #     update
    #     """
    #     self.input_ids_len[bid] = seq_len

    def run(self, model_inputs, **kargs):
        """
        Use ngram_match to get draft tokens from the input and output.
        """
        draft_tokens = model_inputs["draft_tokens"].cpu()
        seq_lens_this_time = kargs["seq_lens_this_time"].cpu()
        seq_lens_encoder = model_inputs["seq_lens_encoder"].cpu()
        seq_lens_decoder = model_inputs["seq_lens_decoder"].cpu()
        print("start run proposer")
        print("before draft_tokens: ", draft_tokens)
        print("pre_ids: ", model_inputs["pre_ids"].cpu())
        ngram_match(
            self.input_ids_cpu,
            self.input_ids_len.cpu(),
            model_inputs["pre_ids"].cpu(),
            model_inputs["step_idx"].cpu(),
            model_inputs["actual_draft_token_num"].cpu(),
            draft_tokens,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            kargs["real_batch_size"],
            self.max_ngram_size,
            self.max_draft_tokens,
        )
        print("after draft_tokens: ", draft_tokens)
        print("seq_lens_this_time: ", seq_lens_this_time)
        print("finish run proposer")

        model_inputs["draft_tokens"][:] = draft_tokens.cuda()
        model_inputs["seq_lens_encoder"][:] = seq_lens_encoder.cuda()
        kargs["seq_lens_this_time"][:] = seq_lens_this_time.cuda()
