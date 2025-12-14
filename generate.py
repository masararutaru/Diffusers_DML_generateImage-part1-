from diffusers import OnnxStableDiffusionPipeline
import numpy as np

# モデルのID（今回は軽量で動作確認しやすい v1.4 のONNX版を使います）
model_id = "CompVis/stable-diffusion-v1-4"

print("モデルをダウンロード・読み込み中... (初回は数GBの通信が発生します)")

# DirectML (AMD/Intel GPU) を使う設定でパイプラインを読み込む
pipe = OnnxStableDiffusionPipeline.from_pretrained(
    model_id,
    revision="onnx",
    provider="DmlExecutionProvider"
)

# プロンプト（生成したい画像の指示）
prompt = "cyberpunk city street at night, neon signs, towering skyscrapers, flying vehicles, rain-soaked streets reflecting lights, highly detailed, cinematic lighting"
print(f"画像生成中: {prompt}")

# 生成実行
image = pipe(prompt).images[0]

# 保存
output_filename = "astronaut_mars.png"
image.save(output_filename)
print(f"完了！画像が保存されました: {output_filename}")