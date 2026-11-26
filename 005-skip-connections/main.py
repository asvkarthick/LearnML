# 1. Encoder pass: Collect feature maps from shallow (large) to deep (small)
skip_connections = ["Layer_1_64x64", "Layer_2_32x32", "Layer_3_16x16"]
print("Before reversing:", skip_connections)

# 2. Reverse the list so deepest features come first
skip_connections = skip_connections[::-1]
print("After reversing: ", skip_connections)

# 3. Decoder pass: Step back up from deep (small) to shallow (large)
decoder_steps = ["Up_to_16x16", "Up_to_32x32", "Up_to_64x64"]

print("\nMatching in the decoder:")
for idx in range(len(decoder_steps)):
    enc_feature = skip_connections[idx]
    dec_feature = decoder_steps[idx]
    print(f"Step {idx}: Combining {dec_feature} + {enc_feature}")

