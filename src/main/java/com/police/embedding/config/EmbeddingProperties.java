package com.police.embedding.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.nio.file.Path;

@ConfigurationProperties(prefix = "embedding")
public record EmbeddingProperties(
        String modelName,
        Path modelPath,
        Path tokenizerPath,
        int maxTokens,
        int maxInputs,
        int maxConcurrency,
        int intraOpThreads,
        int dimension,
        boolean warmup,
        long maxRequestBytes,
        int queueTimeoutSeconds,
        String apiKey,
        int gpuId
) {

    public EmbeddingProperties {
        if (modelName == null || modelName.isBlank()) {
            modelName = "Qwen3-Embedding-0.6B";
        }
        if (modelPath == null) {
            modelPath = Path.of("models", "model_int8.onnx");
        }
        if (tokenizerPath == null) {
            tokenizerPath = Path.of("models", "tokenizer.json");
        }
        if (maxTokens <= 0) {
            maxTokens = 1024;
        }
        if (maxInputs <= 0) {
            maxInputs = 64;
        }
        if (maxConcurrency <= 0) {
            maxConcurrency = 4;
        }
        if (dimension <= 0) {
            dimension = 1024;
        }
        if (maxRequestBytes <= 0) {
            maxRequestBytes = 4L * 1024 * 1024;
        }
        if (queueTimeoutSeconds <= 0) {
            queueTimeoutSeconds = 30;
        }
        if (apiKey != null && apiKey.isBlank()) {
            apiKey = null;
        }
        if (gpuId < 0) {
            gpuId = -1;
        }
    }
}
