package com.police.embedding;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class EmbeddingServerApplication {

    public static void main(String[] args) {
        SpringApplication.run(EmbeddingServerApplication.class, args);
    }
}
