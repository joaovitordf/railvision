# RailVision AI — Protocolo SDD de Evolução e Promoção de Modelos

**Documento:** Protocolo Operacional ML v1.0 (WS0)  
**Escopo:** Detecção (`open_top_freight_wagon`), Rastreamento (`ByteTrack` / `BoT-SORT`) e futuras etapas da Fase 5.

---

## 1. Regras Invioláveis de Rigor Experimental

1. **Zero vazamento e congelamento do `test`:**
   - As campanhas `vid_09` e `vid_16` (`camp_vfli_cosne`) permanecem congeladas no split `test`.
   - O vídeo `vid_01` (`vid_01_langer_unbeladener_g_terzug_von_oben_k_l`) permanece reservado exclusivamente como benchmark negativo (`test_negative_only`).
   - É **proibido** mover `vid_01`, `vid_09` ou `vid_16` para `train` ou `val`, bem como ajustar hiperparâmetros, prompts ou *thresholds* com base em resultados de `test`.
2. **Avaliação oficial em `test` somente via `--eval-test`:**
   - `ml/scripts/train_detector.py` avalia por padrão apenas `val` durante treinamento e exploração.
   - A execução com `--eval-test` é permitida **uma única vez** por ciclo no candidato final vencedor em `val` multi-campanha.
   - Se o candidato falhar no Portão G3, qualquer correção subsequente deve ser iterada por expansão de dados/hipóteses em `train` + `val` (e novo benchmark independente quando aplicável), nunca tunando sobre `test`.
3. **Preservação histórica:**
   - Artefatos e relatórios das Fases 3 (`exp01`–`exp03`), 3.1 (`exp04a`–`exp04c`, `open-top-yolo-v1`) e 4 (`phase4-onnx-spike/`, `phase4-case-vid01-4k/`) não devem ser sobrescritos nem removidos.
   - `train_detector.py` bloqueia sobrescrita de experimentos existentes salvo uso explícito de `--overwrite` em testes descartáveis fora das runs oficiais.
4. **Uma hipótese isolada por experimento:**
   - Cada run (`exp05`, `exp06a`, `exp06b`, etc.) altera apenas um fator principal (dados, descongelamento/augmentation, resolução, arquitetura) e registra hiperparâmetros, seed, versão do dataset/manifesto, ambiente e métricas em `eval_summary.json`.

---

## 2. Calibração e Registro Obrigatório por Experimento

Todo experimento candidato de detecção deve registrar em `eval_summary.json`:
- Identidade da run (`experiment_name`, `base_model`, `data_yaml` relativo, `best_weights` relativo) e *checksums*.
- Hardware efetivo (`device`, `gpu_name`, `peak_allocated_vram_mib`, `train_wall_time_s`).
- Métricas em `val` (`Precision`, `Recall`, `mAP50`, `mAP50_95`, `speed_ms_per_image`).
- **Calibração em `val`:** varredura de `conf` (`0.05..0.85` em `val`) registrando o ponto ótimo de `F1` (`val_conf_calibration.best_f1_conf`), evitando cortes arbitrários como `0.25` sem justificativa.

---

## 3. Política de Arquitetura

1. **Linha principal (`exp05`, `exp06`):** `YOLO26n` e `YOLO26s` (baseline verificado na RTX 4060 Ti 8 GB).
2. **Entrada condicional de `RF-DETR` (`exp07`):** somente se a família `YOLO26` atingir platô comprovado em `val` após expansão de dados, ajuste de resolução e *augmentations*; antes do treino comparativo, registrar auditoria de licença (`Apache-2.0` vs. `Roboflow PML 1.0`), VRAM pico e FPS na RTX 4060 Ti.

---

## 4. Critérios Objetivos dos Portões (Gates SDD)

| Portão | Escopo | Critérios obrigatórios de aprovação |
|---|---|---|
| **G0** | Alinhamento SDD + correções críticas ML/Backend/Frontend/Ops | `pip check` limpo; testes unitários ML/Backend/Frontend e `ng build` verdes; paridade ONNX `minIoU >= 0,999` preservada; fluxo real de inspeção e Lab sem truncamento silencioso nem desalinhamento de overlay |
| **G1** | Model Registry, comparação A/B, vídeos próprios e pré-anotação CVAT | Seleção/comparação A/B de checkpoints na UI; registro explícito de vídeos próprios com bloqueio de vazamento; benchmark local `YOLOE` vs. `YOLO-World` concluído em `val`/`vid_04` (sem tocar `test`/`vid_01`); round-trip `preannotate_cvat -> CVAT/validação -> ingestão` testado |
| **G2** | Expansão curada `open-top-v2` (WS2) | `>= 6` campanhas positivas independentes em `train`, `>= 2` campanhas em `val`, `20–25%` *hard negatives*, `0%` vazamento, pranchas QA revisadas, `test` (`vid_09`/`vid_16`) e negativo (`vid_01`) intactos |
| **G3** | Promoção do Detector/Tracker v2 (WS3) | `test mAP50 >= 0,70`; `test mAP50-95 >= 0,45`; `MOTA >= 0,60`; `IDF1 >= 0,65`; `0` falsos positivos em `vid_01` e locos `vid_16`; erro absoluto de contagem `<= ±1` vagão; GPU `>= 15 FPS` em 1080p; paridade ONNX `minIoU >= 0,995`; novo Model Card e relatório publicados |
| **G4** | Estágio 2 Material + Carregamento (WS4) | Macro F1 + curva cobertura/rejeição (`unknown`) para material; QWK + acurácia adjacente ±1 para nível ordinal em teste independente; regra `empty => material=null` garantida; E2E verde |
