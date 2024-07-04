import torch
import torch.nn as nn
import pdb

from ..data.from_tensors import FromTensors
from .sampling import sample
from .expectation import Expectation
from .text_aggregation import aggregate_features
from .WSI_sampler import WSI_sampling

import clip


class SamplePatches(nn.Module):
    """SamplePatches samples from a high resolution image using an attention
    map. The layer expects the following inputs when called `x_low`, `x_high`,
    `attention`. `x_low` corresponds to the low resolution view of the image
    which is used to derive the mapping from low resolution to high. `x_high`
    is the tensor from which we extract patches. `attention` is an attention
    map that is computed from `x_low`.
    Arguments
    ---------
        n_patches: int, how many patches should be sampled
        patch_size: int, the size of the patches to be sampled (squared)
        receptive_field: int, how large is the receptive field of the attention
                         network. It is used to map the attention to high
                         resolution patches.
        replace: bool, whether we should sample with replacement or without
        use_logits: bool, whether of not logits are used in the attention map
    """

    def __init__(self, n_patches, patch_size, receptive_field=0, replace=False,
                 use_logits=False, **kwargs):
        self._n_patches = n_patches
        self._patch_size = (patch_size, patch_size)
        self._receptive_field = receptive_field
        self._replace = replace
        self._use_logits = use_logits

        super(SamplePatches, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """ Legacy function of the pytorch implementation """
        shape_low, shape_high, shape_att = input_shape

        # Figure out the shape of the patches
        patch_shape = (shape_high[1], *self._patch_size)

        patches_shape = (shape_high[0], self._n_patches, *patch_shape)

        # Sampled attention
        att_shape = (shape_high[0], self._n_patches)

        return [patches_shape, att_shape]

    def forward(self, x_low, x_high, attention, WSI):
        sample_space = attention.shape[1:]
        samples, sampled_attention = sample(
            self._n_patches,
            attention,
            sample_space,
            replace=self._replace,
            use_logits=self._use_logits
        )

        # DIY sampler for WSI by lq
        patches = WSI_sampling(samples, WSI, attention)

        # offsets = torch.zeros_like(samples).float()
        # if self._receptive_field > 0:
        #     offsets = offsets + self._receptive_field / 2

        # Get the patches from the high resolution data
        # Make sure that below works
        # x_low = x_low.permute(0, 2, 3, 1)
        # x_high = x_high.permute(0, 2, 3, 1)
        # assert x_low.shape[-1] == x_high.shape[-1], "Channels should be last for now"


        # patches, _ = FromTensors([x_low, x_high], None).patches(
        #     samples,
        #     offsets,
        #     sample_space,
        #     torch.Tensor([x_low.shape[1:-1]]).view(-1) - self._receptive_field,
        #     self._patch_size,
        #     0,
        #     1
        # )

        return [patches, sampled_attention]



class ATSModel(nn.Module):
    """ Attention sampling model that perform the entire process of calculating the
        attention map, sampling the patches, calculating the features of the patches,
        the expectation and classifices the features.
        Arguments
        ---------
        attention_model: pytorch model, that calculated the attention map given a low
                         resolution input image
        feature_model: pytorch model, that takes the patches and calculated features
                       of the patches
        classifier: pytorch model, that can do a classification into the number of
                    classes for the specific problem
        n_patches: int, the number of patches to sample
        patch_size: int, the patch size (squared)
        receptive_field: int, how large is the receptive field of the attention network.
                         It is used to map the attention to high resolution patches.
        replace: bool, if to sample with our without replacment
        use_logts: bool, if to use logits when sampling
    """

    def __init__(self, attention_model, feature_model, classifier, text_encoder, t2a_model, n_patches, patch_size, receptive_field=0,
                 replace=False, use_logits=False):
        super(ATSModel, self).__init__()

        self.attention_model = attention_model
        self.feature_model = feature_model
        self.classifier = classifier
        self.text_encoder = text_encoder
        self.t2a_model = t2a_model

        self.sampler = SamplePatches(n_patches, patch_size, receptive_field, replace, use_logits)
        # self.expectation = Expectation(replace=replace)
        self.projection = nn.Linear(512, 32)

        self.patch_size = patch_size
        self.n_patches = n_patches
        self.text_select_k = 2

    def forward(self, x_low, x_high, text, WSI_name):
        # First we compute our attention map
        attention_map = self.attention_model(x_low)

        # Gen text feature
        text_token = clip.tokenize(text, truncate=True).cuda()
        text_features = self.text_encoder.encode_text(text_token)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        text_attentions = self.t2a_model(text_features_norm)

        # Then we sample patches based on the attention
        patches, sampled_attention = self.sampler(x_low, x_high, attention_map, WSI_name)
        patches = patches.cuda()
        # We compute the features of the sampled patches
        channels = patches.shape[2]
        patches_flat = patches.view(-1, channels, self.patch_size, self.patch_size)
        patch_features = self.feature_model(patches_flat)
        dims = patch_features.shape[-1]
        patch_features = patch_features.view(-1, self.n_patches, dims)    # (batch, sample, fea_dim)

        text_features_proj = self.projection(text_features_norm.float())
        sample_features = aggregate_features(text_features_proj, patch_features, sampled_attention, self.text_select_k)

        # sample_features = self.expectation(patch_features, sampled_attention)

        y = self.classifier(sample_features)

        return y, attention_map, patches, x_low, text_attentions
