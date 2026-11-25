# WS1: Avaliação Local de Professores Open-Vocabulary vs. `exp04c` na RTX 4060 Ti

- **Data:** 2026-11-24
- **Hardware medido:** `NVIDIA GeForce RTX 4060 Ti` (`cuda:0`, 8 GB VRAM) — `torch 2.9.1+cu128`, `ultralytics 8.4.215`
- **Script reproduzível:** [benchmark_open_vocab_preannotators.py](../../ml/scripts/benchmark_open_vocab_preannotators.py)
- **Artefato bruto JSON:** `artifacts/runs/preannotation/ws1_teacher_benchmark.json`
- **Taxa e amostra do experimento:** `24` frames positivos (`63` caixas) em `val` (`vid_15`) + `10` frames negativos em `train` (`vid_04`), com `conf = 0.25`, `iou_nms = 0.45`, IoU de matching = `0.50` — **amostra reduzida e direcional** (não comprova generalização definitiva entre campanhas).
- **Prompts textuais avaliados:** `["open-top freight railcar", "gondola rail car", "hopper railcar", "open top freight wagon"]`
- **Exemplar visual (`YOLOE` VP):** `data/processed/open-top-yolo-v1/train/images/train__camp_coal_tunnel__vid_14__f002398.jpg` (split `train`)

## 1. Isolamento Obrigatório de Benchmarks

Em estrita conformidade com o protocolo WS0/WS1:
- **Avaliação restrita exclusivamente a:**
  - `val`: 24 frames (`63` caixas GT) de `vid_15` (`camp_sishen_saldanha`).
  - Negativos de `train`: 10 frames de `vid_04` (`camp_perm_overpass`, locomotivas/vagões fechados).
- **Conjuntos congelados excluídos por guarda em código (`assert_no_reserved_benchmark_paths`):** `vid_01`, `vid_09`, `vid_16` e `camp_vfli_cosne`. Nenhum dado de `test` ou do benchmark negativo reservado foi lido neste experimento.

## 2. Resultados Medidos na RTX 4060 Ti

| Modelo / Estratégia | `val vid_15` Precision | `val vid_15` Recall | `val vid_15` F1@50 | `val vid_15` AP@50 | Negativos `vid_04` FP (10 frames) | Latência média (ms/frame) | Pico de VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`exp04c_yolo26n`** (PoC local) | **0.7917** | **0.9048** | **0.8444** | **0.8679** | **0** (`0.00`/frame) | **19.01 ms** | **58.92 MB** |
| `yoloe_text` (`yoloe-26s-seg.pt`) | 0.0000 (`0/22`) | 0.0000 | 0.0000 | 0.0000 | 8 (`0.80`/frame) | 30.52 ms | 141.08 MB |
| `yoloe_visual_prompt` (`yoloe-26s-seg.pt`, 1 exemplar `vid_14`) | **1.0000** (`2/2`) | 0.0317 | 0.0615 | 0.0317 | **0** (`0.00`/frame) | 386.34 ms | 445.14 MB |
| `yolov8s-worldv2_text` | 0.0000 (`0/19`) | 0.0000 | 0.0000 | 0.0000 | 13 (`1.30`/frame) | 22.52 ms | 188.74 MB |
| `hybrid` (`exp04c` + `yoloe_vp`, NMS IoU 0.50) | **0.7917** | **0.9048** | **0.8444** | **0.8679** | **0** (`0.00`/frame) | — | — |

## 3. Análise Comparativa: `YOLOE` vs. `YOLO-World` vs. `exp04c`

1. **Qualidade em domínio ferroviário (`vid_15`) e resistência a falsos positivos (`vid_04`):**
   - Modelos zero-shot baseados apenas em texto (`YOLOE text` e `YOLOv8s-Worldv2 text`) falharam completamente no enquadramento rígido por vagão completo em `vid_15` (`0` TPs em IoU >= 0.50, somando `22` e `19` FPs) e geraram alto ruído nos hard negatives de `vid_04` (`0.80` FP/frame no `YOLOE text` vs. `1.30` FP/frame no `YOLOv8s-Worldv2`).
   - Entre os open-vocabulary, **`YOLOE` com prompt visual (`yoloe-26s-seg.pt`)** superou claramente o `YOLO-World`: obteve **100% de precisão (0 FP em `vid_15` e 0 FP em `vid_04`)**, ao passo que `YOLOv8s-Worldv2` gerou 13 FPs em 10 frames negativos. Contudo, com apenas 1 exemplar de outro ângulo (`vid_14`), o recall zero-shot em `vid_15` foi baixo (`3.17%`), exigindo exemplar visual da mesma campanha para melhorar cobertura.
   - O detector especializado **`exp04c_yolo26n`** obteve desempenho amplamente superior (`AP@50 = 0.8679`, `Recall = 0.9048`, `0` FPs em `vid_04`).

2. **Consumo de VRAM e latência na RTX 4060 Ti (8 GB):**
   - Todos os candidatos cabem com folga na RTX 4060 Ti (< 450 MB de pico em inferência): `exp04c` consome apenas `58.92 MB` (`19.01 ms`), `YOLOE text` `141.08 MB` (`30.52 ms`), `YOLOv8s-Worldv2` `188.74 MB` (`22.52 ms`), e `YOLOE visual prompt` `445.14 MB` (`386.34 ms` recomputando features da imagem de referência por chamada).

3. **Facilidade de integração local:**
   - Tanto `YOLOE` quanto `YOLOWorld` executam 100% offline dentro de `ultralytics==8.4.215` e `ml/.venv` (pesos persistidos em `artifacts/models/pretrained/`), sem envio de imagens para serviços externos e sem alterar `torch`/`ultralytics`.

## 4. Limitações do Experimento e Decisão Revisada para o WS2

> [!WARNING]
> **Escopo estatístico limitado:** Este benchmark utilizou apenas **24 frames positivos de validação (`vid_15`, 1 única campanha)** e **10 frames negativos de treino (`vid_04`, 1 única campanha)**. Os resultados orientam a escolha prática de ferramenta local na RTX 4060 Ti, mas **não provam generalização nem superioridade definitiva** para campanhas inéditas.

1. **Adotar `exp04c` como pré-anotador primário (`--engine exp04c`)** para acelerar a rotulagem inicial em campanhas visualmente próximas ao domínio conhecido e ranquear incertezas (`uncertainty_ranking.json`).
2. **Restringir `YOLOE (yoloe-26s-seg.pt)` em modo visual (`--engine yoloe` / `--engine hybrid`) a exemplares dentro da mesma campanha:**
   - Em avaliação zero-shot cruzada (`vid_14` -> `vid_15`), `YOLOE` visual obteve recall de apenas `3.17%`, embora com `0` falsos positivos. Portanto, seu uso recomendado no WS2 é **fornecer 1–3 frames já revisados da própria campanha nova** como referência visual antes de propagar sugestões no restante do clipe.
3. **Tratar `YOLOE` textual e `YOLOv8s-Worldv2` apenas como alternativas experimentais opcionais:**
   - Ambos apresentaram `0` TPs a `IoU >= 0.50` em `vid_15` e alto ruído nos negativos `vid_04` (`0.80` e `1.30` FP/frame), consumindo memória extra (`mobileclip2_b.ts` / `CLIP`) sem retorno prático que justifique mantê-los como fluxo padrão.
4. **Revisão humana 100% obrigatória mantida:** nenhuma saída de [preannotate_cvat.py](../../ml/scripts/preannotate_cvat.py) entra diretamente em treinamento; tudo passa pelo CVAT com `"humanReviewed": false` até confirmação explícita via `--confirm-human-reviewed` em [validate_cvat_exports.py](../../ml/scripts/validate_cvat_exports.py).
