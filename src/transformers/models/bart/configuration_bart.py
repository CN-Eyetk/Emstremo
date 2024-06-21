# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" BART model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

BART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/config.json",
    # See all BART models at https://huggingface.co/models?filter=bart
}


class BartConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BartModel`. It is used to
    instantiate a BART model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BART `facebook/bart-large
    <https://huggingface.co/facebook/bart-large>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the BART model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BartModel` or
            :class:`~transformers.TFBartModel`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        force_bos_token_to_be_generated (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to force BOS token to be generated at step 1 (after ``decoder_start_token_id``), only
            :obj:`True` for `bart-large-cnn`.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        num_labels: (:obj:`int`, `optional`, defaults to 3):
            The number of labels to use in :class:`~transformers.BartForSequenceClassification`.

    Example::

        >>> from transformers import BartModel, BartConfig

        >>> # Initializing a BART facebook/bart-large style configuration
        >>> configuration = BartConfig()

        >>> # Initializing a model from the facebook/bart-large style configuration
        >>> model = BartModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "bart"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        gradient_checkpointing=False,
        force_bos_token_to_be_generated=False,
        use_cache=True,
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        is_encoder_decoder=True,
        decoder_start_token_id=2,
        **kwargs
    ):
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.force_bos_token_to_be_generated = force_bos_token_to_be_generated  # only relevant for CNN
        self.n_emo_out = kwargs["n_emo_out"] if "n_emo_out" in kwargs.keys() else False
        self.use_th_attn = kwargs["use_th_attn"] if "use_th_attn" in kwargs.keys() else False
        self.use_trans_mat = kwargs["use_trans_mat"] if "use_trans_mat" in kwargs.keys() else False
        self.add_emo_cross_attn = kwargs["add_emo_cross_attn"] if "add_emo_cross_attn" in kwargs.keys() else False
        self.prepend = kwargs["prepend"] if "prepend" in kwargs.keys() else False
        self.use_kl = kwargs["use_kl"] if "use_kl" in kwargs.keys() else False
        self.emo_from_eos = kwargs["emo_from_eos"] if "emo_from_eos" in kwargs.keys() else False
        self.emo_from_situ = kwargs["emo_from_situ"] if "emo_from_situ" in kwargs.keys() else False
        self.use_emo_in_dist = kwargs["use_emo_in_dist"] if "use_emo_in_dist" in kwargs.keys() else False
        self.use_emb_prep = kwargs["use_emb_prep"] if "use_emb_prep" in kwargs.keys() else False
        self.use_copy = kwargs["use_copy"] if "use_copy" in kwargs.keys() else False
        self.st_from_eos = kwargs["st_from_eos"] if "st_from_eos" in kwargs.keys() else False
        self.use_st_seq = kwargs["use_st_seq"] if "use_st_seq" in kwargs.keys() else False
        self.lstm_st_seq = kwargs["lstm_st_seq"] if "lstm_st_seq" in kwargs.keys() else False
        self.merge = kwargs["merge"] if "merge" in kwargs.keys() else False
        self.no_fuse = kwargs["no_fuse"] if "no_fuse" in kwargs.keys() else False
        self.stg_use_cat_attn = kwargs["stg_use_cat_attn"] if "stg_use_cat_attn" in kwargs.keys() else False
        self.emo_use_cat_attn = kwargs["emo_use_cat_attn"] if "emo_use_cat_attn" in kwargs.keys() else False
        self.use_role_embed = kwargs["use_role_embed"] if "use_role_embed" in kwargs.keys() else False
        self.use_vad_labels = kwargs["use_vad_labels"] if "use_vad_labels" in kwargs.keys() else False
        self.use_vae = kwargs["use_vae"] if "use_vae" in kwargs.keys() else False
        self.mixed_vae = kwargs["mixed_vae"] if "mixed_vae" in kwargs.keys() else False
        self.latent_dim = kwargs["latent_dim"] if "latent_dim" in kwargs.keys() else False
        self.sample_strat_emb = kwargs["sample_strat_emb"] if "sample_strat_emb" in kwargs.keys() else False
        self.wo_comet = kwargs["wo_comet"] if "wo_comet" in kwargs.keys() else False
        self.wo_stra = kwargs["wo_stra"] if "wo_stra" in kwargs.keys() else False
        self.wo_emo = kwargs["wo_emo"] if "wo_emo" in kwargs.keys() else False
        self.rl_emb_ratio = kwargs["rl_emb_ratio"] if "rl_emb_ratio" in kwargs.keys() else 0.2
        self.vad_emb_ratio = kwargs["vad_emb_ratio"] if "vad_emb_ratio" in kwargs.keys() else 0.2
        self.emo_loss_ratio = kwargs["emo_loss_ratio"] if "emo_loss_ratio" in kwargs.keys() else 1.0
        self.strategy_loss_ratio = kwargs["strategy_loss_ratio"] if "strategy_loss_ratio" in kwargs.keys() else 0.05
        self.emo_out_loss_ratio = kwargs["emo_out_loss_ratio"] if "emo_out_loss_ratio" in kwargs.keys() else 1.0
        self.intensity_vae = kwargs["intensity_vae"] if "intensity_vae" in kwargs.keys() else False
        self.use_situ_in_encoder = kwargs["use_situ_in_encoder"] if "use_situ_in_encoder" in kwargs.keys() else False
        self.use_situ_in_decoder = kwargs["use_situ_in_decoder"] if "use_situ_in_decoder" in kwargs.keys() else False
        self.use_contrastive_loss = kwargs["use_contrastive_loss"] if "use_contrastive_loss" in kwargs.keys() else False
        self.contrastive_loss_ratio = kwargs["contrastive_loss_ratio"] if "contrastive_loss_ratio" in kwargs.keys() else 0.05
        self.sample_strategy_embedding = kwargs["sample_strategy_embedding"] if "sample_strategy_embedding" in kwargs.keys() else False
        self.fuse_z = kwargs["fuse_z"] if "fuse_z" in kwargs.keys() else False
        self.use_centroid_loss = kwargs["use_centroid_loss"] if "use_centroid_loss" in kwargs.keys() else False
        self.use_uncertainty_loss = kwargs["use_uncertainty_loss"] if "use_uncertainty_loss" in kwargs.keys() else False
        self.stop_norm_weight = kwargs["stop_norm_weight"] if "stop_norm_weight" in kwargs.keys() else False
        self.wo_Sresp = kwargs["wo_Sresp"] if "wo_Sresp" in kwargs.keys() else False
    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model
