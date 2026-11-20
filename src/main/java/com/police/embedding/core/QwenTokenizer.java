package com.police.embedding.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Qwen2/Qwen3 系列 Byte-Level BPE 分词器的纯 Java 实现，只依赖 tokenizer.json。
 * 处理顺序与 HuggingFace tokenizers 一致：
 * NFC 规范化 → 按 added_tokens 切分 → Split 正则 → ByteLevel 字节映射 → BPE 合并 → 追加模板特殊 token。
 * 只做编码，不实现 decode 与截断之外的填充策略。
 */
public final class QwenTokenizer {

    /** BPE 合并对的分隔符；ByteLevel 映射后的字符不会是 NUL，可安全用于拼 key。 */
    private static final char PAIR_SEP = '\u0000';

    private static final String DEFAULT_SPLIT_REGEX =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
                    + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static final int WORD_CACHE_LIMIT = 50_000;

    private final Map<String, Integer> vocab;
    private final Map<String, Integer> mergeRanks;
    private final Pattern splitPattern;
    private final List<String> addedTokens;
    private final Map<String, Integer> addedIds;
    private final int[] appendIds;
    private final int[] byteChars;
    private final Map<String, int[]> wordCache = new ConcurrentHashMap<>();

    private QwenTokenizer(Map<String, Integer> vocab,
                          Map<String, Integer> mergeRanks,
                          Pattern splitPattern,
                          List<String> addedTokens,
                          Map<String, Integer> addedIds,
                          int[] appendIds,
                          int[] byteChars) {
        this.vocab = vocab;
        this.mergeRanks = mergeRanks;
        this.splitPattern = splitPattern;
        this.addedTokens = addedTokens;
        this.addedIds = addedIds;
        this.appendIds = appendIds;
        this.byteChars = byteChars;
    }

    public static QwenTokenizer load(Path tokenizerJson) throws IOException {
        JsonNode root = new ObjectMapper().readTree(tokenizerJson.toFile());

        Map<String, Integer> vocab = new HashMap<>(200_000);
        for (Map.Entry<String, JsonNode> entry : root.path("model").path("vocab").properties()) {
            vocab.put(entry.getKey(), entry.getValue().asInt());
        }

        Map<String, Integer> mergeRanks = new HashMap<>(200_000);
        int rank = 0;
        for (JsonNode merge : root.path("model").path("merges")) {
            String left;
            String right;
            if (merge.isArray()) {
                left = merge.get(0).asText();
                right = merge.get(1).asText();
            } else {
                // 老格式："Ġ t"
                String text = merge.asText();
                int space = text.indexOf(' ');
                left = text.substring(0, space);
                right = text.substring(space + 1);
            }
            mergeRanks.put(left + PAIR_SEP + right, rank++);
        }

        String regex = findSplitRegex(root.path("pre_tokenizer"));
        // HuggingFace 的 Rust 正则按 Unicode 语义处理 \s、\d、\w；
        // Java 默认只认 ASCII，必须开启 UNICODE_CHARACTER_CLASS 才能对齐（例如 NBSP、全角空格）。
        Pattern splitPattern = Pattern.compile(regex == null ? DEFAULT_SPLIT_REGEX : regex,
                Pattern.UNICODE_CHARACTER_CLASS);

        List<String> addedTokens = new ArrayList<>();
        Map<String, Integer> addedIds = new HashMap<>();
        for (JsonNode token : root.path("added_tokens")) {
            String content = token.path("content").asText();
            addedTokens.add(content);
            addedIds.put(content, token.path("id").asInt());
        }
        // 最长匹配优先，避免 <|endoftext|> 被更短的 token 抢先切走
        addedTokens.sort(Comparator.comparingInt(String::length).reversed());

        int[] appendIds = findAppendIds(root.path("post_processor"));

        return new QwenTokenizer(vocab, mergeRanks, splitPattern, addedTokens, addedIds,
                appendIds, buildByteChars());
    }

    /** 编码一条文本，maxTokens 为最终长度上限（含模板追加的特殊 token）。 */
    public long[] encode(String text, int maxTokens) {
        String normalized = Normalizer.normalize(text == null ? "" : text, Normalizer.Form.NFC);
        List<Integer> ids = new ArrayList<>();
        StringBuilder plain = new StringBuilder();
        int i = 0;
        while (i < normalized.length()) {
            String special = matchSpecial(normalized, i);
            if (special != null) {
                encodePlain(plain.toString(), ids);
                plain.setLength(0);
                ids.add(addedIds.get(special));
                i += special.length();
            } else {
                plain.append(normalized.charAt(i++));
            }
        }
        encodePlain(plain.toString(), ids);

        int limit = Math.max(0, maxTokens - appendIds.length);
        int size = Math.min(ids.size(), limit);
        long[] out = new long[size + appendIds.length];
        for (int k = 0; k < size; k++) {
            out[k] = ids.get(k);
        }
        for (int k = 0; k < appendIds.length; k++) {
            out[size + k] = appendIds[k];
        }
        return out;
    }

    private String matchSpecial(String text, int pos) {
        for (String token : addedTokens) {
            if (text.startsWith(token, pos)) {
                return token;
            }
        }
        return null;
    }

    private void encodePlain(String text, List<Integer> ids) {
        if (text.isEmpty()) {
            return;
        }
        Matcher matcher = splitPattern.matcher(text);
        int last = 0;
        while (matcher.find()) {
            if (matcher.start() > last) {
                encodeWord(text.substring(last, matcher.start()), ids);
            }
            encodeWord(matcher.group(), ids);
            last = matcher.end();
        }
        if (last < text.length()) {
            encodeWord(text.substring(last), ids);
        }
    }

    private void encodeWord(String word, List<Integer> ids) {
        for (int id : bpe(toByteLevel(word))) {
            ids.add(id);
        }
    }

    private String toByteLevel(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        StringBuilder sb = new StringBuilder(bytes.length);
        for (byte b : bytes) {
            sb.append((char) byteChars[b & 0xFF]);
        }
        return sb.toString();
    }

    private int[] bpe(String word) {
        int[] cached = wordCache.get(word);
        if (cached != null) {
            return cached;
        }

        List<String> symbols = new ArrayList<>(word.length());
        for (int i = 0; i < word.length(); i++) {
            symbols.add(word.substring(i, i + 1));
        }

        // 每轮取 rank 最小的相邻对合并，与 HuggingFace BPE 的合并顺序一致
        while (symbols.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int bestPos = -1;
            for (int i = 0; i < symbols.size() - 1; i++) {
                Integer rank = mergeRanks.get(symbols.get(i) + PAIR_SEP + symbols.get(i + 1));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestPos = i;
                }
            }
            if (bestPos < 0) {
                break;
            }
            symbols.set(bestPos, symbols.get(bestPos) + symbols.get(bestPos + 1));
            symbols.remove(bestPos + 1);
        }

        int[] ids = new int[symbols.size()];
        for (int i = 0; i < ids.length; i++) {
            Integer id = vocab.get(symbols.get(i));
            if (id == null) {
                throw new IllegalStateException("词表中不存在 BPE 符号: " + symbols.get(i));
            }
            ids[i] = id;
        }
        if (wordCache.size() < WORD_CACHE_LIMIT) {
            wordCache.putIfAbsent(word, ids);
        }
        return ids;
    }

    /** GPT-2 的 bytes_to_unicode 映射表。 */
    private static int[] buildByteChars() {
        int[] map = new int[256];
        int extra = 0;
        for (int b = 0; b < 256; b++) {
            boolean printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            map[b] = printable ? b : 256 + extra++;
        }
        return map;
    }

    private static String findSplitRegex(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        if ("Split".equals(node.path("type").asText())) {
            return node.path("pattern").path("Regex").asText(null);
        }
        for (JsonNode child : node) {
            String regex = findSplitRegex(child);
            if (regex != null) {
                return regex;
            }
        }
        return null;
    }

    /** 读取 post_processor 中 TemplateProcessing 在单句末尾追加的特殊 token id。 */
    private static int[] findAppendIds(JsonNode postProcessor) {
        JsonNode template = findType(postProcessor, "TemplateProcessing");
        if (template == null) {
            return new int[0];
        }
        JsonNode specialTokens = template.path("special_tokens");
        List<Integer> ids = new ArrayList<>();
        for (JsonNode item : template.path("single")) {
            JsonNode special = item.path("SpecialToken");
            if (special.isMissingNode()) {
                continue;
            }
            JsonNode tokenIds = specialTokens.path(special.path("id").asText()).path("ids");
            if (tokenIds.isArray() && !tokenIds.isEmpty()) {
                ids.add(tokenIds.get(0).asInt());
            }
        }
        int[] out = new int[ids.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ids.get(i);
        }
        return out;
    }

    private static JsonNode findType(JsonNode node, String type) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        if (type.equals(node.path("type").asText())) {
            return node;
        }
        for (JsonNode child : node) {
            JsonNode found = findType(child, type);
            if (found != null) {
                return found;
            }
        }
        return null;
    }
}
