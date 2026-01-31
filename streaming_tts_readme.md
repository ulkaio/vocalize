python streaming_tts.py "Long text..." --first-chunk-chars 80 --stream-chars 220 | ffplay -autoexit -nodisp -

python streaming_tts.py "Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects." --first-chunk-chars 40 --stream-chars 220 --speed 1.5 --model-path models/kokoro/kokoro-v1.0.int8.onnx --voices-path models/kokoro/voices-v1.0.bin | ffplay -autoexit -nodisp -


# curl

  curl -s -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input":"Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.", "voice":"af_heart", "stream":true, "first_chunk_chars":50, "stream_chunk_chars":220, "speed": 1.25}' \
    | ffplay -autoexit -nodisp -