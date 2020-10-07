# coding=utf-8
# Copyleft 2019 project LXRT.

import torch
import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder, convert_sents_to_features
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH,
            mode='lxr' # JRS
        )
        hid_dim = self.lxrt_encoder.dim

        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        if 0:

            x = self.lxrt_encoder(sent, (feat, pos))
            logit = self.logit_fc(x)

        else:
            # JRS expanding lxrt_encoder.forward
            train_features = convert_sents_to_features(
                sent, self.lxrt_encoder.max_seq_length, self.lxrt_encoder.tokenizer)

            # JRS from what I can tell, segmend_ids is just 0 and input_mask is just 1 everywhere.
            input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
            input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
            segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

            maxIters = 3
            # ATM this code only works for batch size 1
            assert feat.shape[0] == 1

            count = 0
            for i in range(maxIters):
                count += 1
                ftrTuple, output = self.lxrt_encoder.model(input_ids, segment_ids, input_mask,
                                    visual_feats=(feat, pos),
                                    visual_attention_mask=None)
                lang_feats, visn_feats = ftrTuple
                #print('input_ids shape = {}, feat shape = {}, visn_feats = {}, lang_feats = {}'.format(\
                #       input_ids.shape, feat.shape, visn_feats.shape, lang_feats.shape))
                logit = self.logit_fc(output)
                smax = nn.Softmax(dim=-1)(logit)
                # Find top two values of output for each example.
                topVals = torch.topk(smax, 2).values
                #print('topVals = {}'.format(topVals))
                # If the largest is big by a long way, then we don't need to keep going.
                if topVals[0,1] < topVals[0,0]*0.5:
                    break
                # Prepare for next iteration.  Output vision and language vectors become subsequent inputs.
                feat = visn_feats # after this, internally the downscale layer not invoked
                # The way this works is: later on if input_ids is 2-dim then it embeds it, otherwise treats it
                # as an embedding.
                input_ids = lang_feats

            #print('VQAModel::forward: iters = {}/{}'.format(count, maxIters))


        return logit
