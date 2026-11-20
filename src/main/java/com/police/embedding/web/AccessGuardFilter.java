package com.police.embedding.web;

import com.police.embedding.config.EmbeddingProperties;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/**
 * /v1/embeddings 的入口保护：可选 Bearer API Key 校验 + 请求体大小限制。
 * 在 MVC 反序列化之前执行，避免超大 JSON 先被读进内存。
 */
@Component
public class AccessGuardFilter extends OncePerRequestFilter {

    private final EmbeddingProperties props;

    public AccessGuardFilter(EmbeddingProperties props) {
        this.props = props;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {
        if ("/v1/embeddings".equals(request.getRequestURI())
                && "POST".equalsIgnoreCase(request.getMethod())) {
            String apiKey = props.apiKey();
            if (apiKey != null && !validKey(request.getHeader("Authorization"), apiKey)) {
                writeError(response, 401, "无效的 API Key", "authentication_error");
                return;
            }
            if (request.getContentLengthLong() > props.maxRequestBytes()) {
                writeError(response, 413, "请求体超过上限 " + props.maxRequestBytes() + " 字节",
                        "invalid_request_error");
                return;
            }
        }
        chain.doFilter(request, response);
    }

    private static boolean validKey(String authorization, String apiKey) {
        if (authorization == null) {
            return false;
        }
        byte[] expected = ("Bearer " + apiKey).getBytes(StandardCharsets.UTF_8);
        byte[] actual = authorization.getBytes(StandardCharsets.UTF_8);
        return MessageDigest.isEqual(expected, actual);
    }

    private static void writeError(HttpServletResponse response, int status, String message, String type)
            throws IOException {
        response.setStatus(status);
        response.setCharacterEncoding("UTF-8");
        response.setContentType("application/json");
        response.getWriter().write("{\"error\":{\"message\":\"" + message
                + "\",\"type\":\"" + type + "\",\"code\":null}}");
    }
}
