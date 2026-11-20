# Relatório de Experimentos — Fase 3.1 (Validação Inicial Especializada em `open_top_freight_wagon`)

> [!IMPORTANT]
> **Aviso explícito de escopo (Validação Inicial Multi-Campanha):** Os resultados deste relatório constituem uma **validação inicial especializada** da classe única `open_top_freight_wagon` sobre `4` campanhas públicas (`141` imagens de detecção e `61` quadros de tracking). **Eles NÃO demonstram generalização para produção**, pois o treino conta com apenas **`1` vídeo/geometria positiva (`vid_14`, carvão nos EUA)** complementado por quadros negativos sem caixa positiva em `vid_04`. A **Fase 5** (materiais e níveis de carregamento) permanece **pendente**. Todos os *checkpoints* e resultados da Fase 3 (`exp01`, `exp02`, `exp03` e `phase3_video_tracking_eval.json`) foram **preservados intactos**.

---

## 1. Ambiente, GPU Utilizada e Dataset Anotado Manualmente

- **Ambiente de execução (`eval_summary.json` e `phase3_1_open_top_mot_eval.json`):**
  - Python `3.12.8` · `torch==2.9.1+cu128` · `ultralytics==8.4.215` · `lap==0.5.14`
  - **Dispositivo efetivo em todos os treinos e benchmarks de tracking:** **`cuda:0` (`NVIDIA GeForce RTX 4060 Ti`, `8188 MiB` VRAM, `cuda_available=true`)**.
- **Dataset manual ingerido e auditado (`data/processed/open-top-yolo-v1`, manifesto `specs/data-manifests/open-top-split-v1.json`):**
  - Separação estrita por campanha (**zero vazamento**, com `vid_09` e `vid_16` juntas em `camp_vfli_cosne`, e **nenhum frame** da Tarefa 2 de tracking incluído no treino):

| Split | Campanhas / Vídeos | Imagens totais | Imagens c/ caixa | Imagens vazias (`0` caixas / *hard negatives*) | Caixas `open_top_freight_wagon` | Papel |
|---|---|---|---|---|---|---|
| **`train`** | `camp_coal_tunnel` (`vid_14`: `72`) + `camp_perm_overpass` (`vid_04`: `10`) | **`82`** | `64` (`vid_14`) | `18` (`8` loco `vid_14` + `10` neve/fechados `vid_04`) | **`502`** (`vid_14=502`, `vid_04=0`) | *Fine-tuning* (apenas `1` geometria positiva real: gôndolas/hoppers de carvão `vid_14`). |
| **`val`** | `camp_sishen_saldanha` (`vid_15`) | **`24`** | `19` | `5` (locomotivas de cauda) | **`63`** | Seleção de *checkpoint* sem vazamento (gôndolas altas de minério). |
| **`test` (Detecção)** | `camp_vfli_cosne` (`vid_09`: `25` + `vid_16`: `10`) | **`35`** | `30` (`vid_09=25`, `vid_16=5`) | `5` (locomotivas acopladas `vid_16`) | **`93`** (`vid_09=88`, `vid_16=5`) | Medir transferência para *hoppers* europeus inéditos com brita cinza-clara. |
| **Total Detecção** | **`4` campanhas** | **`141`** | **`113`** | **`28`** *(arquivos `.txt` vazios gravados)* | **`658`** | — |
| **`test` (Tracking GT)** | `camp_vfli_cosne` (`vid_09`, `t=12,0s..24,0s`, frames `300..600` passo `5`) | **`61`** *(a `5 fps`)* | `61` | `0` | **`181`** (`8` IDs contínuos `1..8`, **`0` gaps**) | Referência humana **exclusiva de avaliação** (`MOT 1.1`) em `5 fps`. |

---

## 2. Experimentos de Detecção (`open_top_freight_wagon` · `cuda:0`)

Parâmetros base: `imgsz=640`, `batch=8`, `seed=42`, `deterministic=true`. O *checkpoint* vencedor da Fase 3.1 foi escolhido **exclusivamente pelas métricas de `val`** (`mAP50` como critério primário alinhado ao `IoU=0,50` do benchmark MOT; nunca usando `test` para seleção).

| Experimento | Inicialização / Estratégia | Épocas / Tempo / VRAM (`cuda:0`) | `val P / R / mAP50 / mAP50-95` (`model.val()`) | `val` inf. (`ms`) | `test P / R / mAP50 / mAP50-95` (`model.val()`) | `test TP / FP / FN` (`conf=0,25, IoU=0,50`) |
|---|---|---|---|---|---|---|
| **exp04a** (`exp04a_yolo26n_opentop_from_coco`) | `yolo26n.pt` (COCO) / FT completo (`AdamW lr0=0,002` auto) | `35/80` *(early stop)* · `154,63s` · `1419,5 MiB` | `0,7958 / 0,6508 / 0,7729 / 0,3141` | `10,665` | `0,2616 / 0,1600 / 0,0912 / 0,0190` | `7 / 26 / 86` (`micro_P=0,2121`, `micro_R=0,0753`) |
| **exp04b** (`exp04b_yolo26n_opentop_from_twist_exp02`) | `exp02/best.pt` (TWIST) / FT completo (`AdamW lr0=0,002` auto) | `36/80` *(early stop)* · `154,67s` · `1410,8 MiB` | `0,6883 / 0,5079 / 0,6119 / 0,3272` | `11,009` | `0,0055 / 0,6237 / 0,0081 / 0,0017` | `0 / 0 / 93` (nenhuma detecção `>= 0,25`) |
| **exp04c** *(Selecionado em `val`)* (`exp04c_yolo26n_opentop_freeze10`) | `yolo26n.pt` (COCO) / **`freeze=10` + `lr0=0,0005` + `cos_lr=true`** | **`60/60`** · `205,22s` · **`893,7 MiB`** | **`0,8394 / 0,9127 / 0,9218 / 0,6266`** | `8,805` | **`0,4520 / 0,2688 / 0,2558 / 0,0772`** | `3 / 3 / 90` (`micro_P=0,5000`, `micro_R=0,0323`) |

### Diagnóstico das curvas de treinamento e análise de erro
1. **Colapso catastrófico em `exp04a` / `exp04b` e cura pela regularização em `exp04c`:**
   - Em `exp04a` e `exp04b`, atualizar toda a rede com `lr0=0,002` em apenas `82` imagens fez a validação atingir o pico nas épocas `10–11` (`0,77485` e `0,62506` em `results.csv`) e **desabar para `~0,0005` entre as épocas `14` e `26`**, decorando a textura exata de `vid_14`.
   - Congelar o *backbone* (`freeze=10`) com passo reduzido (`lr0=0,0005`, decaimento cosseno) em `exp04c` **eliminou o colapso**: a curva cresceu de forma estável até `val mAP50 = 0,92264` (época `24`) e **`val mAP50 = 0,9218 / mAP50-95 = 0,6266`** na avaliação final do `best.pt` — dobrando o `mAP50-95` na campanha inédita `vid_15` com **`893,7 MiB`** de VRAM — e elevou `test mAP50` de `0,0912` (`exp04a`) e `0,0081` (`exp04b`) para **`0,2558`**.
2. **Por que o *recall* e a confiança no `test` (`vid_09` + `vid_16`) continuaram baixos (`test mAP50=0,2558` e `TP=3` em `conf=0,25`):**
   - **Inspeção visual de `exp04a/b/c_test_errors.jpg`:**
     - `train` (`vid_14`) e `val` (`vid_15`) compartilham **gôndolas minerais retangulares escuras/fechadas** com chassi baixo e bordas paralelas uniformes — razão pela qual `exp04c` generalizou bem para `vid_15` (`val mAP50=0,9218`).
     - Já o `test` (`vid_09` + `vid_16`) apresenta ***hoppers* europeus em "V" com topo cinza-claro (brita), longarina/chassi aberto visível por baixo e truques salientes**. Nas imagens de `exp04a`, o modelo chega a localizar os vagões em primeiro plano (`conf=0,39..0,74`), mas desenha caixas apertadas apenas na caçamba superior enquanto o GT manual cobre o vagão completo com chassi e engates (`IoU ~0,35–0,45 < 0,50`, punindo a caixa como FP e FN simultaneamente). Já `exp04c`, devido aos pesos do *backbone* congelado + `18` fundos negativos no treino, emite escores de confiança majoritariamente abaixo de `0,25` nesse domínio visual inédito (`mAP50=0,2558` na curva PR integral, porém apenas `6` caixas acima de `0,25`, com `0` falsos positivos em locomotivas `vid_16`).

---

## 3. Benchmark de Rastreamento Anotado (`vid_09`, `61` quadros a `5 fps`, `cuda:0`)

> [!NOTE]
> **Escopo das métricas MOT:** Conforme estabelecido antes da anotação, todas as métricas abaixo (`MOTA`, `IDF1`, `IDSW`, `FM`, `Det P/R`) são calculadas **estritamente sobre os `61` quadros amostrados a `5 fps`** (`vid_09`, frames nativos `300..600` passo `5`, `t=12,0s..24,0s`, `181` caixas GT e `8` identidades contínuas `1..8`), e **não** sobre todos os frames nativos de `25 fps`. Todas as execuções rodaram em **`cuda:0` (`NVIDIA GeForce RTX 4060 Ti`)** em `conf=0,25`. O modo `native25_eval5` roda o *tracker* frame a frame a `25 fps` (`301` quadros) e avalia sobre os `61` instantes exatos; o modo `direct5` roda diretamente sobre os `61` quadros a `5 fps`.

| Execução | Modelo | Tracker / Fluxo | `Det TP / FP / FN` (`IoU=0,50`) | `Det P / R` | `MOTA` (`IoU=0,50`) | `IDF1` | `IDSW` | `FM` | Diagnóstico `IoU=0,30` (`TP / MOTA / IDF1`) | FPS efetivo / VRAM (`cuda:0`) |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline Fase 3 (`exp02`) | `exp02` (`TWIST`, 4 classes*) | `ByteTrack` (`25->5 fps`) | `31 / 1 / 150` | `0,9688 / 0,1713` | `0,1602` | `0,2817` | `1` | `5` | `32 / 0,1713 / 0,2911` | `26,27 fps` (`11,46s`) / `60,0 MiB` |
| Baseline Fase 3 (`exp02`) | `exp02` (`TWIST`, 4 classes*) | `BoT-SORT` (`25->5 fps`) | `32 / 0 / 149` | `1,0000 / 0,1768` | `0,1713` | `0,2911` | `1` | `4` | `32 / 0,1713 / 0,2911` | `21,08 fps` (`14,28s`) / `64,4 MiB` |
| **Fase 3.1 (`exp04c`)** | `exp04c` (`open_top`, `freeze=10`) | `ByteTrack` (`25->5 fps`) | `6 / 3 / 175` | `0,6667 / 0,0331` | `0,0166` | `0,0632` | **`0`** | `1` | `9 / 0,0497 / 0,0947` | **`33,63 fps`** (`8,95s`) / `55,3 MiB` |
| **Fase 3.1 (`exp04c`)** | `exp04c` (`open_top`, `freeze=10`) | `BoT-SORT` (`25->5 fps`) | `7 / 2 / 174` | `0,7778 / 0,0387` | **`0,0276`** | **`0,0737`** | **`0`** | `1` | `9 / 0,0497 / 0,0947` | `21,81 fps` (`13,80s`) / `61,4 MiB` |
| **Fase 3.1 (`exp04c`)** | `exp04c` (`open_top`, `freeze=10`) | `ByteTrack` (`5 fps` direto) | `5 / 0 / 176` | `1,0000 / 0,0276` | `0,0276` | `0,0215` | `0` | `0` | `5 / 0,0276 / 0,0215` | `24,96 fps` (`2,44s`) / `69,3 MiB` |
| **Fase 3.1 (`exp04c`)** | `exp04c` (`open_top`, `freeze=10`) | `BoT-SORT` (`5 fps` direto) | `5 / 0 / 176` | `1,0000 / 0,0276` | `0,0276` | `0,0215` | `0` | `0` | `5 / 0,0276 / 0,0215` | `19,92 fps` (`3,06s`) / `55,3 MiB` |

*\*Observação crítica sobre o baseline `exp02`:* na avaliação agnóstica de classe, o `exp02` atingiu `MOTA=0,1602–0,1713` porque algumas caixas erradas de **`high_speed_train`** que cobriam o vagão central no trecho `300..600` tiveram `IoU >= 0,50` com o GT central (`32` de `181` caixas), ainda que com classe errada.

### Síntese comparativa `ByteTrack` vs. `BoT-SORT` e taxa de quadros
1. **`BoT-SORT` vs. `ByteTrack`:** em `25->5 fps`, o `BoT-SORT` obteve associação ligeiramente superior ao `ByteTrack` nos dois detectores (`exp04c`: `TP=7` vs. `6`, `IDF1=0,0737` vs. `0,0632`, `MOTA=0,0276` vs. `0,0166`; `exp02`: `TP=32` vs. `31`, `FM=4` vs. `5`, `MOTA=0,1713` vs. `0,1602`), mas operando ~35% mais devagar (`21,81 fps` vs. `33,63 fps`).
2. **`25 fps` vs. `5 fps` direto no tracker:** rodar a inferência/associação a `25 fps` e avaliar nos `61` frames gerou `9` predições contra apenas `5` no modo `5 fps` direto (`IDF1=0,0737` vs. `0,0215`), pois os passos menores de movimento entre frames nativos mantiveram trilhas de baixa confiança vivas na interpolação do filtro de Kalman (`diagnostic_iou30: 9/9` `TP`, `det_precision=1,0000` em `exp04c`).
3. **Inspeção visual (`phase3_1_vid09_*_samples.jpg`):** quando o `exp04c` detecta em `vid_09` (por ex. `mot=49/61` em `25 fps`), a caixa predita encaixa corretamente no *hopper* individual sem confundir com locomotiva (`mean_matched_iou` entre `0,6405` e `0,7361`), mas o *recall* abaixo de `5%` em `conf=0,25` mostra que o gargalo **não está no algoritmo de tracking**, e sim na **falta de exemplos de *hoppers* europeus no conjunto de treino** (`train` teve `1` única campanha positiva: `vid_14`).

---

## 4. Limitações Documentadas e Próximo Passo

1. **Treino com apenas 1 campanha positiva real (`vid_14`):** `vid_04` recebeu `0` caixas anotadas (atuando apenas como negativos de neve/vagões fechados). Com `train` contendo só gôndolas retangulares americanas de carvão, `exp04c` alcançou `0,9218 mAP50` em gôndolas similares (`vid_15` em `val`), mas não generalizou em `conf=0,25` para *hoppers* europeus com brita clara e chassi exposto (`vid_09`/`vid_16`).
2. **Validação inicial, não produção:** o experimento valida o *pipeline* ponta a ponta CVAT -> treino regularizado -> métricas MOT reais em GPU, deixando claro que futura expansão exigirá múltiplas campanhas positivas no treino (incluindo *hoppers* em "V" e ângulos variados).
3. **Pronto para a Fase 4 (Spike Java/ONNX — Portão P2):** conforme o plano aprovado, o próximo passo técnico é exportar o modelo `YOLO26n` (`exp04c` e/ou referência leve) para `.onnx` e validar **ONNX Runtime em Java 21 + decodificação de vídeo no Windows** (comparando contra o fallback Python FastAPI) para decidir a arquitetura de produção atrás de `InferenceEngine`.
4. **Fase 5 (materiais e carregamento):** permanece **pendente**, seguindo a busca por vídeos públicos licenciados que mostrem a superfície de minério de ferro, calcário, areia e carvão sem forçar classes sem dados representativos.

## Artefatos locais associados (fora do Git)
- Peso selecionado na Fase 3.1 (`val mAP50=0,9218`): `artifacts/runs/detect/exp04c_yolo26n_opentop_freeze10/weights/best.pt`
- JSONs de avaliação: `artifacts/runs/detect/exp04a/b/c.../eval_summary.json` e `artifacts/runs/tracking/phase3_1_open_top_mot_eval.json`
- Pranchas inspecionadas: `data/processed/triage-frames/qa_detection_{train,val,test}.jpg`, `qa_tracking_vid09_5fps.jpg`, `exp04a/b/c*_test_errors.jpg` e `phase3_1_vid09_*_samples.jpg`
- Baselines preservados da Fase 3: `artifacts/runs/detect/exp01..03/weights/best.pt` e `artifacts/runs/tracking/phase3_video_tracking_eval.json`
