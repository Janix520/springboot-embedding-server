package com.police.embedding.core;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import com.police.embedding.config.EmbeddingProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Qwen3-Embedding 的 ONNX 推理入口：左填充批量编码 → session.run → 取末 token 隐藏状态 → 截断维度 → L2 归一化。
 */
@Service
public class EmbeddingEngine implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(EmbeddingEngine.class);

    private final EmbeddingProperties props;
    private final QwenTokenizer tokenizer;
    private final OrtEnvironment environment;
    private final OrtSession session;
    private final Semaphore permits;
    private final String outputName;
    private final int hiddenSize;
    private final boolean hasPositionIds;
    private final List<CacheInput> cacheInputs;

    public EmbeddingEngine(EmbeddingProperties props) throws IOException, OrtException {
        this.props = props;

        Path modelPath = props.modelPath().toAbsolutePath();
        Path tokenizerPath = props.tokenizerPath().toAbsolutePath();
        if (!Files.isRegularFile(modelPath)) {
            throw new IllegalStateException("模型文件不存在: " + modelPath);
        }
        if (!Files.isRegularFile(tokenizerPath)) {
            throw new IllegalStateException("tokenizer.json 不存在: " + tokenizerPath);
        }

        this.tokenizer = QwenTokenizer.load(tokenizerPath);
        this.permits = new Semaphore(Math.max(1, props.maxConcurrency()));
        this.environment = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(props.intraOpThreads());
        options.setInterOpNumThreads(1);
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        if (props.gpuId() >= 0) {
            options.addCUDA(props.gpuId());
            log.info("推理设备: CUDA:{}", props.gpuId());
        } else {
            log.info("推理设备: CPU");
        }

        long started = System.currentTimeMillis();
        this.session = environment.createSession(modelPath.toString(), options);
        long loadMillis = System.currentTimeMillis() - started;

        Set<String> inputs = session.getInputNames();
        if (!inputs.contains("input_ids") || !inputs.contains("attention_mask")) {
            throw new IllegalStateException("模型输入不符合预期: " + inputs + "，需要 input_ids/attention_mask");
        }
        this.hasPositionIds = inputs.contains("position_ids");
        this.cacheInputs = collectCacheInputs(inputs);

        Set<String> outputs = session.getOutputNames();
        this.outputName = outputs.contains("last_hidden_state")
                ? "last_hidden_state" : outputs.iterator().next();
        long[] outputShape = ((TensorInfo) session.getOutputInfo().get(outputName).getInfo()).getShape();
        this.hiddenSize = outputShape.length == 3 && outputShape[2] > 0
                ? (int) outputShape[2] : props.dimension();

        log.info("模型加载完成: {} ({} ms)，输入 {} 个（KV cache {} 组），输出 {}[hidden={}]",
                modelPath.getFileName(), loadMillis, inputs.size(), cacheInputs.size() / 2,
                outputName, hiddenSize);

        if (props.warmup()) {
            long warmStart = System.currentTimeMillis();
            embed(List.of("warmup"), null, null);
            log.info("预热完成: {} ms", System.currentTimeMillis() - warmStart);
        }
    }

    /** 输出维度即模型隐藏层维度。 */
    public int dimension() {
        return hiddenSize;
    }

    /**
     * 编码一组文本。instruction 非空时按 Qwen3-Embedding 官方格式拼成 "Instruct: ...\nQuery:..."；
     * dimensions 非空时按 Matryoshka 截断后重新归一化。
     *
     * 注意：该 ONNX 图在 batch>1 时存在跨样本注意力串扰（两条无关文本一起推理，
     * 各自向量也会漂移），因此必须逐条推理。
     */
    public Result embed(List<String> texts, String instruction, Integer dimensions) {
        int dim = dimensions == null ? hiddenSize : Math.min(dimensions, hiddenSize);
        List<float[]> vectors = new ArrayList<>(texts.size());
        int tokens = 0;
        for (String text : texts) {
            long[] ids = tokenizer.encode(withInstruction(text, instruction), props.maxTokens());
            tokens += ids.length;
            vectors.add(run(ids, dim));
        }
        return new Result(vectors, tokens, dim);
    }

    private float[] run(long[] ids, int dim) {
        int seqLen = ids.length;
        long[] shape = {1, seqLen};
        long[] attentionMask = new long[seqLen];
        long[] positionIds = new long[seqLen];
        Arrays.fill(attentionMask, 1);
        for (int i = 0; i < seqLen; i++) {
            positionIds[i] = i;
        }

        List<OnnxTensor> emptyCaches = new ArrayList<>(cacheInputs.size());
        try (OnnxTensor idsTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(ids), shape);
             OnnxTensor maskTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(attentionMask), shape);
             OnnxTensor posTensor = hasPositionIds
                     ? OnnxTensor.createTensor(environment, LongBuffer.wrap(positionIds), shape) : null) {

            Map<String, OnnxTensor> feeds = new HashMap<>();
            feeds.put("input_ids", idsTensor);
            feeds.put("attention_mask", maskTensor);
            if (hasPositionIds) {
                feeds.put("position_ids", posTensor);
            }
            // 模型图带 KV cache 输入；传空缓存走无缓存前向，输出里的 present.* 丢弃
            for (CacheInput cache : cacheInputs) {
                OnnxTensor empty = OnnxTensor.createTensor(environment, ByteBuffer.allocateDirect(0),
                        new long[]{1, cache.kvHeads(), 0, cache.headDim()}, cache.type());
                emptyCaches.add(empty);
                feeds.put(cache.name(), empty);
            }

            if (!acquirePermit()) {
                throw new BusyException("服务繁忙，请稍后重试");
            }
            try (OrtSession.Result result = session.run(feeds)) {
                OnnxTensor hidden = (OnnxTensor) result.get(outputName).orElseGet(() -> result.get(0));
                return readVector(hidden, seqLen - 1, dim);
            } finally {
                permits.release();
            }
        } catch (OrtException e) {
            throw new IllegalStateException("ONNX 推理失败: " + e.getMessage(), e);
        } finally {
            for (OnnxTensor tensor : emptyCaches) {
                try {
                    tensor.close();
                } catch (Exception ignored) {
                    // 释放失败不影响结果
                }
            }
        }
    }

    /** 取指定位置的隐藏状态；同时兼容 fp32 与 fp16 输出，dimensions 截断在 L2 归一化之前完成。 */
    private float[] readVector(OnnxTensor hidden, int lastToken, int dim) {
        long[] shape = hidden.getInfo().getShape();
        int hiddenDim = (int) shape[2];
        int size = Math.min(dim, hiddenDim);
        int base = lastToken * hiddenDim;
        OnnxJavaType type = hidden.getInfo().type;

        float[] vector = new float[size];
        double squareSum = 0;
        if (type == OnnxJavaType.FLOAT) {
            FloatBuffer buffer = hidden.getFloatBuffer();
            for (int k = 0; k < size; k++) {
                float value = buffer.get(base + k);
                vector[k] = value;
                squareSum += (double) value * value;
            }
        } else if (type == OnnxJavaType.FLOAT16) {
            ShortBuffer buffer = hidden.getShortBuffer();
            for (int k = 0; k < size; k++) {
                float value = Float.float16ToFloat(buffer.get(base + k));
                vector[k] = value;
                squareSum += (double) value * value;
            }
        } else {
            throw new IllegalStateException("不支持的模型输出类型: " + type);
        }

        if (squareSum > 0) {
            float scale = (float) (1.0 / Math.sqrt(squareSum));
            for (int k = 0; k < size; k++) {
                vector[k] *= scale;
            }
        }
        return vector;
    }

    private boolean acquirePermit() {
        try {
            return permits.tryAcquire(props.queueTimeoutSeconds(), TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("推理等待被中断", e);
        }
    }

    /** 收集 KV cache 输入的固定维度，运行时补形状为 [batch, kv_heads, 0, head_dim] 的空张量。 */
    private List<CacheInput> collectCacheInputs(Set<String> inputs) throws OrtException {
        Map<String, NodeInfo> inputInfo = session.getInputInfo();
        List<CacheInput> list = new ArrayList<>();
        for (String name : inputs) {
            if (!name.startsWith("past_key_values.")) {
                continue;
            }
            TensorInfo info = (TensorInfo) inputInfo.get(name).getInfo();
            long[] shape = info.getShape();
            if (shape.length != 4 || shape[1] <= 0 || shape[3] <= 0) {
                throw new IllegalStateException("无法识别 KV cache 输入形状: " + name + " " + Arrays.toString(shape));
            }
            list.add(new CacheInput(name, shape[1], shape[3], info.type));
        }
        return list;
    }

    private static String withInstruction(String text, String instruction) {
        if (instruction == null || instruction.isBlank()) {
            return text;
        }
        return "Instruct: " + instruction.strip() + "\nQuery:" + text;
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (Exception e) {
            log.warn("关闭 ONNX Session 失败", e);
        }
    }

    private record CacheInput(String name, long kvHeads, long headDim, OnnxJavaType type) {
    }

    /** 排队超过 queue-timeout-seconds 时抛出，由 Controller 映射为 503。 */
    public static class BusyException extends RuntimeException {
        public BusyException(String message) {
            super(message);
        }
    }

    public record Result(List<float[]> vectors, int tokens, int dimension) {
    }
}
