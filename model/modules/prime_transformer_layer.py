from fairseq.modules import TransformerEncoderLayer, TransformerDecoderLayer
from .prime_mha import MultiheadAttention820


class PrimeTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, layer_id):
        self.layer_id = layer_id
        super().__init__(args)
        
    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention820(
            self.embed_dim, args.encoder_attention_heads, layer_id=self.layer_id, args=args,
            dropout=args.attention_dropout, cur_attn_type='es'
        )


class PrimeTransformerDecoderLayer(TransformerDecoderLayer):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, layer_id, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        self.layer_id = layer_id
        super().__init__(args, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)

    def build_self_attention(
            self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention820(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            layer_id=self.layer_id,
            args=args,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            cur_attn_type='ds'
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention820(
            self.embed_dim, args.decoder_attention_heads,
            layer_id=self.layer_id,
            args=args,
            dropout=args.attention_dropout,
            cur_attn_type='dc',
        )

