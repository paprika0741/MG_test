import torch
import os
import torch.distributed as dist
from megatron.core.transformer.moe.moe_utils  import topk_softmax_with_capacity

class PredictorController:
    def __init__(self, step, path, config):
        # indexed-1
        self.config = config
        self.step = step
        predictor_num = self.config.num_layers - step
        self.predictors = [None] * (predictor_num + 1)
    
        for l in range(1,predictor_num+1 ):
            target_layer = l + step 
            if target_layer > self.config.num_layers:
                continue
            predictor = torch.nn.Parameter(torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32))
            predictor.data = predictor.data.to(dtype=config.params_dtype)
            # copy from  megatron/core/transformer/moe/router.py
            self.enable_expert_bias = self.config.moe_router_enable_expert_bias
            if self.enable_expert_bias:
                self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32),
                persistent=False,
                )
                self.register_buffer(
                    'expert_bias', torch.zeros(self.config.num_moe_experts, dtype=torch.float32)
                )
            else:
                self.local_tokens_per_expert = None
                self.expert_bias = None
            
            weight_path = os.path.join(path, f'decoder_layers_{target_layer - 1}_mlp_router_weight.pt')
            print(f"RANK [{dist.get_rank()}] load predictor at layer {l}, target layer = {target_layer}, from path {weight_path}")
            weight = torch.load(weight_path, map_location="cpu")
            weight = weight.to(dtype= predictor.dtype, device= predictor.device)
            with torch.no_grad():
                predictor.copy_(weight)
            self.predictors [l] = predictor
        

        # indexed-1
        self.pred_routing_maps = [None] * (predictor_num + 1)
        # indexed-1
        self.predictor_streams = [torch.cuda.Stream() for _ in range(predictor_num + 1)]
        assert len(self.predictors) == predictor_num + 1
        assert len(self.pred_routing_maps) == predictor_num + 1
        assert len(self.predictor_streams) == predictor_num + 1
    def submit(self, layer_id, hidden_state: torch.Tensor):
        hidden = hidden_state.detach()
        def async_predict():
            with torch.cuda.stream(self.predictor_streams[layer_id]):
                predictor = self.predictors[layer_id]
                self.pred_routing_maps[layer_id] =  self.predict(hidden, predictor)   
        async_predict()

    def get_routing_map(self, layer_id):
        torch.cuda.current_stream().wait_stream(self.predictor_streams[layer_id])
        return self.pred_routing_maps[layer_id]
    def predict(self, input: torch.Tensor, predictor):
        if predictor.device.type == 'cpu':
            predictor.data = predictor.data.to(device=torch.cuda.current_device())
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        logits = torch.nn.functional.linear(input.to(router_dtype), predictor.to(router_dtype))
        pred_nex_layer_output = logits.view(-1,logits.shape[-1])
        _, pred_routing_map, _ = topk_softmax_with_capacity(
            pred_nex_layer_output,
            self.config.moe_router_topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.config.moe_router_score_function,
            expert_bias=self.expert_bias,
        )
        return pred_routing_map
