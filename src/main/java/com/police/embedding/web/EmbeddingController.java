package com.police.embedding.web;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.police.embedding.config.EmbeddingProperties;
import com.police.embedding.core.EmbeddingEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.HttpMediaTypeNotSupportedException;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * OpenAI 兼容的 /v1/embeddings 接口。
 * 在标准字段之外支持可选的 instruction（Qwen3-Embedding 检索指令）与 dimensions（Matryoshka 截断）。
 */
@RestController
public class EmbeddingController {

    private static final Logger log = LoggerFactory.getLogger(EmbeddingController.class);

    private final EmbeddingEngine engine;
    private final EmbeddingProperties props;

    public EmbeddingController(EmbeddingEngine engine, EmbeddingProperties props) {
        this.engine = engine;
        this.props = props;
    }

    /** 存活与模型状态检查，供负载均衡/容器探针使用。 */
    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("status", "ok");
        body.put("model", props.modelName());
        body.put("dimension", engine.dimension());
        return body;
    }

    @PostMapping("/v1/embeddings")
    public EmbeddingResponse embeddings(@RequestBody EmbeddingRequest request) {
        List<String> texts = parseInput(request.input());
        if (texts.size() > props.maxInputs()) {
            throw new IllegalArgumentException("input 数量超过上限 " + props.maxInputs());
        }

        Integer dimensions = request.dimensions();
        if (dimensions != null && (dimensions < 1 || dimensions > engine.dimension())) {
            throw new IllegalArgumentException(
                    "dimensions 必须在 1~" + engine.dimension() + " 之间");
        }

        String format = request.encodingFormat() == null
                ? "float" : request.encodingFormat().toLowerCase(Locale.ROOT);
        if (!"float".equals(format) && !"base64".equals(format)) {
            throw new IllegalArgumentException("encoding_format 仅支持 float 或 base64");
        }

        EmbeddingEngine.Result result = engine.embed(texts, request.instruction(), dimensions);
        List<Item> data = new ArrayList<>(result.vectors().size());
        for (int i = 0; i < result.vectors().size(); i++) {
            float[] vector = result.vectors().get(i);
            data.add(new Item("embedding", i, "base64".equals(format) ? toBase64(vector) : vector));
        }

        String model = request.model() == null || request.model().isBlank()
                ? props.modelName() : request.model();
        return new EmbeddingResponse("list", data, model,
                new Usage(result.tokens(), result.tokens()));
    }

    private static List<String> parseInput(JsonNode input) {
        if (input == null || input.isNull()) {
            throw new IllegalArgumentException("input 不能为空");
        }
        if (input.isTextual()) {
            return List.of(input.asText());
        }
        if (input.isArray()) {
            List<String> texts = new ArrayList<>(input.size());
            for (JsonNode node : input) {
                if (!node.isTextual()) {
                    throw new IllegalArgumentException("input 数组只能包含字符串（暂不支持 token id）");
                }
                texts.add(node.asText());
            }
            if (texts.isEmpty()) {
                throw new IllegalArgumentException("input 不能是空数组");
            }
            return texts;
        }
        throw new IllegalArgumentException("input 必须是字符串或字符串数组");
    }

    private static String toBase64(float[] vector) {
        ByteBuffer buffer = ByteBuffer.allocate(vector.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : vector) {
            buffer.putFloat(value);
        }
        return Base64.getEncoder().encodeToString(buffer.array());
    }

    @ExceptionHandler(IllegalArgumentException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Map<String, Object> handleBadRequest(IllegalArgumentException e) {
        return error(e.getMessage(), "invalid_request_error");
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Map<String, Object> handleUnreadable(HttpMessageNotReadableException e) {
        return error("请求体不是合法的 JSON", "invalid_request_error");
    }

    @ExceptionHandler(HttpMediaTypeNotSupportedException.class)
    @ResponseStatus(HttpStatus.UNSUPPORTED_MEDIA_TYPE)
    public Map<String, Object> handleMediaType(HttpMediaTypeNotSupportedException e) {
        return error("Content-Type 必须是 application/json", "invalid_request_error");
    }

    @ExceptionHandler(HttpRequestMethodNotSupportedException.class)
    @ResponseStatus(HttpStatus.METHOD_NOT_ALLOWED)
    public Map<String, Object> handleMethod(HttpRequestMethodNotSupportedException e) {
        return error("仅支持 POST /v1/embeddings", "invalid_request_error");
    }

    @ExceptionHandler(EmbeddingEngine.BusyException.class)
    @ResponseStatus(HttpStatus.SERVICE_UNAVAILABLE)
    public Map<String, Object> handleBusy(EmbeddingEngine.BusyException e) {
        log.warn("推理排队超时，返回 503");
        return error(e.getMessage(), "rate_limit_error");
    }

    @ExceptionHandler(Exception.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public Map<String, Object> handleServerError(Exception e) {
        // 细节只写日志，响应不暴露内部路径与堆栈信息
        log.error("推理失败", e);
        return error("内部推理错误", "server_error");
    }

    private static Map<String, Object> error(String message, String type) {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("message", message);
        error.put("type", type);
        error.put("code", null);
        return Map.of("error", error);
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record EmbeddingRequest(
            String model,
            JsonNode input,
            @JsonProperty("encoding_format") String encodingFormat,
            Integer dimensions,
            String instruction) {
    }

    public record EmbeddingResponse(
            String object,
            List<Item> data,
            String model,
            Usage usage) {
    }

    public record Item(String object, int index, Object embedding) {
    }

    public record Usage(
            @JsonProperty("prompt_tokens") int promptTokens,
            @JsonProperty("total_tokens") int totalTokens) {
    }
}
