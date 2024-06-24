from transformers.models.qwen2 import Qwen2Model,Qwen2Config
# 假设我们有一个Qwen2Model实例
config = Qwen2Config()
model = Qwen2Model(config)

# 提取第四层
fourth_layer = model.layers[3]

print(fourth_layer)
