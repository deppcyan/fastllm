#include <torch/extension.h>

#include "mlp.h"
#include "rotary_embedding.h"
#include "attention.h"

class DecoderLayer {
    public:
    DecoderLayer();

    torch::Tensor forward();

    private:
    MLPLayer mlp_layer_;
    AttentionLayer attn_layer_;
    RotaryEmbeddingLayer rotary_embedding_layer_;
};