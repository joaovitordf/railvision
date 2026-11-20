# ADR-0002: Portão P2 — Decisão pelo Runtime Embutido Java 21 + ONNX Runtime + JavaCV/FFmpeg

- **Status:** Aceito (Portão P2 Aprovado com ressalvas operacionais documentadas)
- **Data:** 2026-11-20
- **Referências:** [ADR-0001](ADR-0001-modular-monolith-and-inference-boundary.md), [Relatório do Spike da Fase 4](../experiments/phase4-java-onnx-spike-report.md), `artifacts/runs/phase4-onnx-spike/java_onnx_parity.json`, `artifacts/runs/phase4-onnx-spike/java_onnx_benchmark.json`

## Contexto

O [ADR-0001](ADR-0001-modular-monolith-and-inference-boundary.md) condicionou a arquitetura definitiva de inferência do RailVision AI ao **Portão P2** na Fase 4: executar um *spike* reproduzível no hardware Windows alvo (`AMD Ryzen 7 5700X3D`, `32 GB RAM`, `RTX 4060 Ti 8 GB`, Amazon Corretto `21.0.5`) comparando o runtime embutido **Java 21 + ONNX Runtime + decodificação de vídeo** contra a referência em **PyTorch / Python** antes de decidir entre manter inferência em Java ou adotar o fallback `ml/serving/` (FastAPI).

Para eliminar qualquer conflito com experimentos anteriores, o modelo `exp04c_yolo26n_opentop_freeze10/weights/best.pt` foi preservado intacto e exportado de forma isolada para `artifacts/models/onnx/exp04c_yolo26n_opentop_640.onnx` (`9,81 MB`, `opset 17`, tensor `[1, 3, 640, 640] -> [1, 5, 8400]`).

## Critérios Objetivos do Portão P2 e Evidências Medidas

| Critério | Meta Objetiva | Medição Observada | Parecer |
|---|---|---|---|
| **P2-C1: Paridade numérica e geométrica** | Contagem idêntica nos frames de referência, `IoU ≥ 0,98`, erro máximo `≤ 3,0 px`, confiança `≤ 0,01` | `8/8` frames exatos entre PyTorch CPU, ONNX Python e ONNX Java; **`minMatchedIou = 0,99930`**, **`maxCoordDiffPx = 0,0838 px`**, **`maxConfDiff = 0,000749`** (`1,0000` / `0,0 px` nos frames `val`) | **APROVADO** |
| **P2-C2: Decodificação e gravação de vídeo no Windows** | Decodificar `.ogv`/`.webm`/`.mp4` em Java sem depender de `ffmpeg.exe` no `PATH` e gerar saída reproduzível no navegador | `JavaCV 1.5.14` + `ffmpeg 8.1.2-1.5.14` (`windows-x86_64`, LGPL) decodificou `theora`, `vp8` e `vp9`; encoder **`libopenh264` (`mp4`, `yuv420p`, `faststart`)** gerou vídeos anotados com streaming `HTTP 206` | **APROVADO** |
| **P2-C3: Latência e vazão (*throughput*)** | `≥ 5 FPS` ponta a ponta e `p95 ≤ 200 ms/frame` em `1080p` no hardware local | Mesmo em **CPU (`CPUExecutionProvider`)**: **`25,53 FPS`** (`vid_15` `1080p` nativo, `p95 = 32,96 ms`), **`29,97–30,15 FPS`** (`5 fps` em `1080p` Theora/VP9, `p95 = 30,48–32,41 ms`), e **`9,33 FPS`** (`p95 = 111,49 ms`) incluindo gravação completa H.264 `1080p` | **APROVADO** |
| **P2-C4: Consumo de memória** | Working Set estável e `≤ 2.500 MB` em Java | Working Set medido entre **`247 MB` e `397 MB`** (`jvmHeapAfter ≤ 121 MB`) em todos os 6 cenários de benchmark | **APROVADO** |
| **P2-C5: Simplicidade operacional** | Processo único backend Java 21 sem orquestração Python em produção nem alteração de `JAVA_HOME` global | Dependências puramente Maven (`pom.xml`), binários nativos embutidos para Windows x64, zero Python em runtime e reaproveitamento direto do stack Spring Boot + Angular (`.\start.ps1`) | **APROVADO** |

## Decisão

1. **Aprovar o runtime embutido Java 21 + ONNX Runtime + JavaCV/FFmpeg (LGPL)** atrás da fronteira `InferenceEngine` no monólito Spring Boot, dispensando o serviço auxiliar `ml/serving/` (FastAPI) para detecção YOLO26 ONNX e decodificação de vídeo no MVP.
2. **Adotar `CPUExecutionProvider` como baseline explícito e verificado no Windows** neste ambiente:
   - Inspecionado explicitamente em runtime (`OrtEnvironment.getAvailableProviders() = [CPUExecutionProvider, AzureExecutionProvider]`, `cudaAvailable = false`).
   - O host possui `RTX 4060 Ti 8 GB`, porém não possui CUDA Toolkit/cuDNN instalados globalmente no `PATH` do Windows, e o Maven Central não distribui `onnxruntime_directml` oficial para Java. Ainda assim, o throughput medido em CPU no `Ryzen 7 5700X3D` (`25,5–39,5 FPS` ponta a ponta) supera com folga a meta do Portão P2.
   - A arquitetura em `OnnxDetectorSession` mantém detecção transparente de provedores caso `onnxruntime_gpu` + CUDA/cuDNN venham a ser habilitados no futuro, sem nunca presumir CUDA silenciosamente.
3. **Fixar `com.microsoft.onnxruntime:onnxruntime:1.17.3` enquanto Amazon Corretto `21.0.5` for o JDK ativo:**
   - Verificado empiricamente no spike que `onnxruntime:1.32.0` falha com `EXCEPTION_ACCESS_VIOLATION` em `corretto-21.0.5\bin\msvcp140.dll` (`14.29.30153.0`) devido à quebra ABI de `std::mutex` (`_Mtx_init_in_situ`) introduzida no VS 2022 `17.10+` (`msvcp140 >= 14.40`).
   - Como o modelo exportado usa `IR 8 / opset 17`, a versão **`1.17.3`** opera integralmente sem modificar a instalação externa do JDK. Atualizações futuras do ONNX Runtime Java exigem JDK 21 com `msvcp140 >= 14.40`.
4. **Manter a qualidade de detecção (`exp04c`) e o tracking fora desta aprovação de runtime:**
   - Este ADR aprova exclusivamente a **viabilidade técnica da plataforma Java 21 + ONNX Runtime**. O modelo `exp04c` continua **reprovado para produção** em razão da generalização limitada no conjunto de teste (`vid_09`/`vid_16`, `mAP50 = 0,2558`).
   - O tracking multiobjeto (`ByteTrack` / associação por IoU) ainda não foi integrado na camada Java (`trackingIntegrated = false`).

## Gatilhos para Revisitação (Quando Reabrir o Fallback FastAPI)

Reavaliar `ml/serving/` via ADR futuro somente se ocorrer pelo menos um dos cenários abaixo nas Fases 5–6:
- Necessidade de modelos pesados (ex.: segmentação densa, `YOLO26s/m`, ReID ou CMC pesado de câmera no `BoT-SORT`) cuja latência agregada em `CPUExecutionProvider` ultrapasse `200 ms/frame` mesmo em amostragem de `5 fps` e a instalação de CUDA/cuDNN para Java não for desejável;
- Divergência de manutenção caso um algoritmo de pós-processamento ou tracking avançado disponível em Python exija replicação complexa em Java com risco de perda de `IDF1`/`MOTA`.
