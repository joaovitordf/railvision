# Model Card — `yolo26n-open-top-poc` (Fase 3.1)

## 1. Identificação do Modelo
- **Experimento selecionado (por `val mAP50`):** `exp04c_yolo26n_opentop_freeze10`
- **Arquitetura base:** `YOLO26n` (`Ultralytics 8.4.215`, `PyTorch 2.9.1+cu128`, entrada `640x640`, `2.389.735` parâmetros, `freeze=10`, `AdamW lr0=0,0005`, `cos_lr=true`)
- **Artefato local (fora do Git):** `artifacts/runs/detect/exp04c_yolo26n_opentop_freeze10/weights/best.pt`
- **Status:** **Validação inicial multi-campanha (`4` campanhas) — NÃO apto para produção e não generaliza para tipos de vagões ausentes do treino.**

## 2. Classe, Uso Pretendido e Usos Fora de Escopo
- **Classe única (`0`):** `open_top_freight_wagon`.
- **Uso pretendido:** Artefato candidato para o **Spike Java + ONNX Runtime (Fase 4 / Portão P2)** e referência comparativa contra a Fase 3 (`exp02`) na evolução do dataset especializado.
- **Fora de escopo / proibido:**
  - Não usar em operação ferroviária real.
  - Não usar para classificação de material (`coal`, `iron_ore`, `limestone`, `sand`) ou estimativa de carregamento (**Fase 5** segue pendente).
  - Não assumir *recall* alto em vagões *hopper* europeus ou ângulos diferentes de gôndolas minerais retangulares sem expansão do treino.

## 3. Dados de Treinamento e Licenciamento
- **Origem:** Subconjunto anotado manualmente no CVAT (`specs/data-manifests/open-top-split-v1.json`, separação estrita por campanha, tracking GT excluído do treino):
  - `train`: `82` imagens (`502` caixas exclusivamente de `vid_14`, *CC BY 3.0*; `10` imagens de `vid_04`, *CC BY 3.0*, com `0` caixas usadas como negativos em neve/vagões fechados).
  - `val`: `24` imagens (`63` caixas de `vid_15`, *CC BY-SA 3.0*).
  - `test`: `35` imagens esparsas (`93` caixas de `vid_09` + `vid_16`, *CC BY-SA 4.0*) + `61` quadros sequenciais de tracking GT em `vid_09` (`181` caixas, `8` IDs).
- **Nota de licença:** Vídeo `vid_15` (`val`) e `vid_09`/`vid_16` (`test`) usam `CC BY-SA`; pesos treinados exclusivamente sobre `vid_14` (`CC BY 3.0`) + `vid_04` (`CC BY 3.0`) exigem atribuição aos respectivos autores em eventual redistribuição.

## 4. Métricas Quantitativas Medidas (`NVIDIA GeForce RTX 4060 Ti`, `cuda:0`)
- **Validação inédita (`val`, `vid_15`, `24` imagens, `63` caixas):**
  - `Precision = 0,8394` · `Recall = 0,9127` · `mAP50 = 0,9218` · `mAP50-95 = 0,6266`
  - Latência de inferência em `val`: `8,805 ms/img` (`0,598 ms` pré / `1,682 ms` pós); VRAM pico de treino: `893,7 MiB`.
- **Teste inédito esparso (`test`, `vid_09` + `vid_16`, `35` imagens, `93` caixas):**
  - `model.val()` (ponto ótimo `F1`): `Precision = 0,4520` · `Recall = 0,2688` · `mAP50 = 0,2558` · `mAP50-95 = 0,0772`
  - Ponto fixo (`model.predict`, `conf=0,25`, `IoU=0,50`): `TP = 3`, `FP = 3`, `FN = 90` (`micro_P = 0,5000`, `micro_R = 0,0323`).
- **Benchmark Tracking GT (`vid_09`, `61` quadros a `5 fps`, `181` caixas GT, `conf=0,25`, `cuda:0`):**
  - `ByteTrack` (`25 fps` avaliado a `5 fps`): `MOTA = 0,0166`, `IDF1 = 0,0632`, `IDSW = 0`, `FM = 1`, `Det P/R = 0,6667 / 0,0331`, `33,63 fps` (`55,3 MiB` VRAM).
  - `BoT-SORT` (`25 fps` avaliado a `5 fps`): `MOTA = 0,0276`, `IDF1 = 0,0737`, `IDSW = 0`, `FM = 1`, `Det P/R = 0,7778 / 0,0387`, `21,81 fps` (`61,4 MiB` VRAM).
  - *(Em `diagnostic_iou30`, ambos os trackers alcançaram `9/9` acertos `det_precision=1,0000`, `MOTA=0,0497`, `IDF1=0,0947`, confirmando precisão espacial local nas poucas detecções emitidas acima de `0,25`, mas baixíssima cobertura por domínio inédito).*

## 5. Limitações Principais
- **Cobertura positiva restrita a `1` vídeo no treino (`vid_14`):** forte desempenho em gôndolas minerais fechadas semelhantes (`vid_15`), mas queda severa de *recall* em `conf=0,25` para *hoppers* europeus de brita (`vid_09`).
- **Necessidade de calibração / mais campanhas de treino:** inclusão de pelo menos 2–3 campanhas positivas distintas (incluindo *hoppers* em "V") no treino antes de depender do detector como rastreador em produção.
