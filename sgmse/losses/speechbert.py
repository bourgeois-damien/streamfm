import torch
from torchaudio.transforms import Resample
from transformers import HubertModel, Wav2Vec2Model, WavLMModel, AutoModel

from .shared import Loss


class SpeechBERTLoss(Loss):
    """
    Based on https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics/calculate_speechbert_score.py

    Reference-Aware Automatic Evaluation of Speech Generation Leveraging NLP Evaluation Metrics
        https://arxiv.org/abs/2401.16812

    The paper suggests to use the precision:
        >While the original BERTScore defines precision, recall and F1-score, we use the precision
        >as we found that it performed the best in our preliminary experiment (Appendix A).
    This implementation by default uses which_score='precision', but allows to choose between
    'precision', 'recall', or 'f1_score'.

    LICENSE copied from https://github.com/urgent-challenge/urgent2025_challenge repo:

                                     Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
    """

    @property
    def domain(self):
        return 'time'

    @property
    def name(self):
        return "speechbert"

    def __init__(
        self,
        sampling_rate: int,  # input sampling rate, will resample to 16 kHz if needed
        which_score: str = "precision",  # 'precision', 'recall', or 'f1'. The paper proposes 'precision' as the best
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.resampler = Resample(self.sampling_rate, 16000, lowpass_filter_width=256)
        self.speech_bert_score = SpeechBERTScore(
            sr=16000, model_type="mhubert-147", layer=8,
        )
        self.which_score = which_score
        assert which_score in ["precision", "recall", "f1"], \
            f"Invalid which_score: {which_score}. Choose from 'precision', 'recall', or 'f1'."

    def forward(self, x: torch.Tensor, xhat: torch.Tensor):
        assert x.shape == xhat.shape
        assert x.ndim == 3  # shape: (B, F, T)

        x, xhat = self.resampler(x), self.resampler(xhat)
        precision, recall, f1_score = self.speech_bert_score.score(x, xhat)  # API is (reference, degraded)

        if self.which_score == 'precision':
            return 1 - precision.mean()
        elif self.which_score == 'recall':
            return 1 - recall.mean()
        elif self.which_score == 'f1':
            return 1 - f1_score.mean()


class SpeechBERTScore(torch.nn.Module):
    """
    Adapted from discrete_speech_metrics package
    # Copyright 2024 Takaaki Saeki
    # MIT LICENSE (https://opensource.org/license/mit/)
    """

    def __init__(self, sr=16000, model_type="hubert-base", layer=None):
        """
        Args:
            sr (int): Sampling rate.
            model_type (str): Model type. Select from "hubert-base", "hubert-large", "wav2vec2-base", "wav2vec2-large", "wavlm-base", "wavlm-base-plus", "wavlm-large".
            layer (int): Layer number to extract features. If None, the last layer is used.
            use_gpu (bool): Whether to use GPU.
        """
        super().__init__()

        if model_type == "hubert-base":
            self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        elif model_type == "hubert-large":
            self.model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")
        elif model_type == "wav2vec2-base":
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        elif model_type == "wav2vec2-large":
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large")
        elif model_type == "wavlm-base":
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        elif model_type == "wavlm-base-plus":
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        elif model_type == "wavlm-large":
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-large")
        elif model_type == "mhubert-147":
            # some warnings may appear depending on the environment but should be fine given the discussion below
            # https://huggingface.co/utter-project/mHuBERT-147/discussions/7
            self.model = AutoModel.from_pretrained('utter-project/mHuBERT-147')
        else:
            raise ValueError(f"Not found the setting for {model_type}.")

        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.layer = layer
        self.sr = sr
        self.resampler = Resample(orig_freq=sr, new_freq=16000, lowpass_filter_width=256)

    def process_feats(self, audio):
        """
        Args:
            audio (torch.Tensor): Audio waveform tensor (1, T).
        """
        if self.layer == None:
            feats = self.model(audio).last_hidden_state
        else:
            feats_hiddens = self.model(audio, output_hidden_states=True).hidden_states
            feats = feats_hiddens[self.layer]
        return feats

    def score(self, gt_wav, gen_wav):
        """
        Args:
            gt_wav (torch.Tensor): Ground truth waveforms (B, T).
            gen_wav (torch.Tensor): Generated waveform (B, T).
        Returns:
            float: Precision.
            float: Recall.
            float: F1 score.
        """
        if gt_wav.ndim == 3:
            gt_wav = gt_wav.squeeze(1)
            assert gt_wav.ndim == 2, "expected single-channel input"
        if gen_wav.ndim == 3:
            gen_wav = gen_wav.squeeze(1)
            assert gen_wav.ndim == 2, "expected single-channel input"

        if self.sr != 16000:  # FIXME not sure about this one?!
            gt_wav = self.resampler(gt_wav)
            gen_wav = self.resampler(gen_wav)

        v_ref = self.process_feats(gt_wav)
        v_gen = self.process_feats(gen_wav)
        precision, recall, f1_score = batched_bert_score(v_gen, v_ref)

        return precision, recall, f1_score


def _bert_score(v_generated, v_reference):
    """
    Args:
        v_generated (torch.Tensor): Generated feature tensor (T, D).
        v_reference (torch.Tensor): Reference feature tensor (T, D).
    Returns:
        float: Precision.
        float: Recall.
        float: F1 score.
    """
    assert v_generated.ndim == 2
    assert v_reference.ndim == 2

    # Calculate cosine similarity
    sim_matrix = torch.matmul(v_generated, v_reference.T) / (torch.norm(v_generated, dim=1, keepdim=True) * torch.norm(v_reference, dim=1).unsqueeze(0))

    # Calculate precision and recall
    precision = torch.max(sim_matrix, dim=1)[0].mean()
    recall = torch.max(sim_matrix, dim=0)[0].mean()

    # Calculate F1 score
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score


# Thanks to Danilo
batched_bert_score = torch.vmap(_bert_score, in_dims=0, out_dims=0, randomness='different')
