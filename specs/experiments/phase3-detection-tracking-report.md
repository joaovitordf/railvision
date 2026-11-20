# Relatório de Experimentos — Fase 3 (Prova de Conceito Camada C+: Detecção e Rastreamento)

> [!IMPORTANT]
> **Aviso explícito de escopo (Prova de Conceito):** Os resultados documentados neste relatório constituem exclusivamente uma **prova de conceito (PoC)** para detecção de material rodante e rastreamento temporário no domínio de treinamento (`TWIST`). **Eles NÃO demonstram generalização para vagões abertos (`open_top_freight_wagon`) nem para operação em produção**, **NÃO** classificam subtipo aberto, material transportado ou nível de carregamento, e evidenciam quantitativamente o *domain shift* severo quando um modelo treinado em câmera fixa rente ao solo (`TWIST`) é aplicado a câmeras oblíquas elevadas (`vid_09`, `vid_05`, `vid_04`). Todos os *checkpoints* (`exp01`, `exp02`, `exp03`) e o registro `phase3_video_tracking_eval.json` são **preservados intactos** como referência pré-especialização para a **Fase 3.1**.

## 1. Ambiente, Reprodutibilidade e Governança de Dados

- **Hardware:** `NVIDIA GeForce RTX 4060 Ti` (`8188 MiB` VRAM).
- **Stack ML (`ml/pyproject.toml`):** Python `3.12.8`, `torch==2.9.1+cu128`, `torchvision==0.24.1+cu128`, `ultralytics==8.4.215`, `lap==0.5.14`, `opencv-python==5.1.0.95`, `pillow==13.0.0`, `numpy==2.5.5`.
- **Hiperparâmetros fixos de treinamento (`ml/scripts/train_detector.py`):** `epochs=40`, `patience=15`, `imgsz=640`, `batch=16`, `seed=42`, `deterministic=true`, AMP ativo, `workers=2`.
- **Política anti-vazamento (`ml/scripts/prepare_twist_dataset.py` e `specs/data-manifests/twist-split-v1.json`):**
  - Origem: `TWIST detailed_labels` (`1.492` imagens brutas em câmera fixa única de inverno; `2` linhas de anotação com área zero descartadas; `8` arquivos duplicados exatos por SHA-256 removidos antes do *split*).
  - **Agrupamento estrito por evento de passagem** (intervalo temporal `> 120s` entre timestamps dos nomes de arquivo): `46` eventos independentes divididos em **`train = 31`** (`3` de carga), **`val = 8`** (`1` de carga: evento `23`, `2023-12-14T19:50:04..12`, `72` imagens) e **`test = 7`** (`1` de carga: evento `15`, `2023-12-11T08:36:54..08:37:04`, `21` imagens). Nenhuma imagem de um mesmo evento atravessa splits.
  - **Mapeamento conservador (`4` classes):** `0: locomotive`, `1: passenger_wagon`, `2: freight_car` (jamais equiparada automaticamente a `freight_wagon_open` / `open_top_freight_wagon`), `3: high_speed_train`. Removidas `TWIST 0 (train)` e `TWIST 6 (object — placa amarela em primeiro plano)`.
  - **`vid_09`, `vid_05`, `vid_04` não vistos no treino; `vid_07` excluído:** Nenhum frame dos vídeos públicos foi incluído no treinamento. O `vid_07` (ponte do rio Kyu-Omono, vista lateral de contêineres) foi excluído de acordo com a resolução da contradição na auditoria (`specs/dataset-audit.md`).

### Tabela de *Splits* e Ablação de Near-Duplicates

| Dataset (`data.yaml`) | Regra de filtragem | `train` imgs (vazias) | `train` instâncias (`loco / pass / freight / hst`) | `val` imgs | `test` imgs |
|---|---|---|---|---|---|
| `data/processed/twist-yolo-v1` | Remoção apenas das `8` duplicatas exatas (SHA-256) | `1.058` (`328`) | `138 / 680 / 570 / 187` (`1.575` total) | `265` (`72` vazias; `11/214/67/56`) | `161` (`12` vazias; `15/187/107/52`) |
| `data/processed/twist-yolo-v1-dedup-near` | Mesmas `val` e `test`; removidos `360` frames próximos no **treino** (`dHash <= 2` intra-evento) | `698` (`150`) | `103 / 608 / 510 / 92` (`1.313` total) | `265` (idêntica) | `161` (idêntica) |

---

## 2. Experimentos de Detecção (`TWIST detailed_labels`)

### Resultados comparativos (`val` para seleção; `test` retido como diagnóstico)

| Experimento | Modelo / Split | Épocas / Tempo / VRAM | `val P / R / mAP50 / mAP50-95` | `val freight_car (P / R / mAP50)` | `val inf (ms)` | `test P / R / mAP50 / mAP50-95` | `test freight_car (mAP50)` | `test TP / FP / FN` (`conf=0,25, IoU=0,50`) |
|---|---|---|---|---|---|---|---|---|
| **exp01** (`exp01_yolo26n_twist_exactdedup`) | `YOLO26n` / `twist-yolo-v1` | `40/40` · `630,57s` · `2516,4 MiB` | `0,6777 / 0,4877 / 0,5876 / 0,4714` | `0,6675 / 0,6716 / 0,7719` | `2,836` | `0,7928 / 0,6215 / 0,6644 / 0,5051` | `0,4645` (`R=0,3832`) | `135 / 22 / 226` |
| **exp02** (`exp02_yolo26n_twist_neardedup_ablation`) *(Selecionado em `val`)* | `YOLO26n` / `twist-yolo-v1-dedup-near` | `40/40` · `458,65s` · `2541,9 MiB` | `0,6717 / 0,6549 / 0,6720 / 0,5031` | `0,8901 / 0,8806 / 0,9337` | `2,420` | `0,7827 / 0,4697 / 0,5679 / 0,4038` | `0,3842` (`R=0,3084`) | `75 / 19 / 286` |
| **exp03** (`exp03_yolo26s_twist`) | `YOLO26s` (`9,95M` params) / `twist-yolo-v1-dedup-near` | `36/40` · `571,49s` · `4572,8 MiB` | `0,6860 / 0,6416 / 0,6528 / 0,4797` | `0,9108 / 0,7623 / 0,8616` | `5,263` | `0,6663 / 0,5525 / 0,6216 / 0,4604` | `0,5978` (`R=0,5234`) | `67 / 8 / 294` |

### Esclarecimento técnico: divergência entre `test_metrics` (`P=0,7827 / R=0,4697`) e `TP=75 / FP=19 / FN=286` no `exp02`

Uma leitura superficial poderia esperar que `TP=75 / (75+286) = 0,2078` coincidisse com `test R=0,4697`. Auditoria direta sobre o mesmo *checkpoint* (`exp02/weights/best.pt`) e conjunto `test` (`161` imagens, `361` caixas GT) confirmou **três causas simultâneas e mensuráveis** para essa diferença:

1. **Diferença de limiar de confiança (*threshold*):** `model.val(split="test")` varre predições a partir de `conf=0,001` e reporta `box.mp / box.mr` próximos ao **ponto de máximo `F1` médio** na curva PR suavizada (`conf ≈ 0,0300`, onde o `F1` médio bruto é `0,6179`). Já `build_test_error_sheet()` avalia em um **ponto operacional fixo `conf=0,25`**, descartando muitas detecções corretas de confiança baixa entre `0,03` e `0,25` (por exemplo: `high_speed_train` tem `39` predições em `conf >= 0,03` contra apenas `15` em `conf >= 0,25`; `passenger_wagon` cai de `163` para `47`). Na própria curva de `val()`, em `conf=0,2503`, a média macro já cai para `P=0,8499 / R=0,3342`.
2. **Agregação *macro* (por classe) vs. *micro* (por instância):** `0,7827 / 0,4697` são médias **macro** igualmente ponderadas (`25%` por classe), elevadas por `high_speed_train` (`52` instâncias, `R=0,7976` no `val()`) e `locomotive` (`15` instâncias, `R=0,4468`). Já `TP=75 / FP=19 / FN=286` são totais **micro** sobre todas as `361` instâncias GT, dominadas por `passenger_wagon` (`187/361 = 51,8%`) e `freight_car` (`107/361 = 29,6%`).
3. **Diferença de *pipeline* de inferência:** `model.val()` usa lote com *letterbox* retangular (`rect=True`) e suavização monotônica na curva `ap_per_class`, enquanto `build_test_error_sheet()` usa `model.predict(conf=0,25)` imagem a imagem sem suavização.

#### Detalhamento medido do `exp02` em ponto fixo (`model.predict`, `conf=0,25`, `IoU=0,50`, conjunto `test`)

| Classe | `GT` | `TP` | `FP` | `FN` | Precision (`conf=0,25`) | Recall (`conf=0,25`) | Referência `model.val()` (ponto ótimo `F1`) |
|---|---|---|---|---|---|---|---|
| `0: locomotive` | `15` | `4` | `0` | `11` | `1,0000` | `0,2667` | `P=0,7695 / R=0,4468` |
| `1: passenger_wagon` | `187` | `28` | `19` | `159` | `0,5957` | `0,1497` | `P=0,5485 / R=0,3262` |
| `2: freight_car` | `107` | `28` | `0` | `79` | `1,0000` | `0,2617` | `P=0,8802 / R=0,3084` |
| `3: high_speed_train` | `52` | `15` | `0` | `37` | `1,0000` | `0,2885` | `P=0,9325 / R=0,7976` |
| **Total Micro (`conf=0,25`)** | **`361`** | **`75`** | **`19`** | **`286`** | **`micro_P = 0,7979`** | **`micro_R = 0,2078`** | — |
| **Média Macro (`conf=0,25`)** | — | — | — | — | **`macro_P = 0,8989`** | **`macro_R = 0,2416`** | **`P=0,7827 / R=0,4697`** |

### Achados técnicos principais dos detectores
1. **Impacto real da ablação de near-duplicates (`exp01` vs. `exp02`):**
   - Na **validação (`val`)**, remover `360` frames semelhantes do treino reduziu o tempo de treino em `27,3%` (`630,57s -> 458,65s`) e elevou **`mAP50-95` (`0,4714 -> 0,5031`)**, **`mAP50` (`0,5876 -> 0,6720`)** e **`freight_car mAP50` (`0,7719 -> 0,9337`)**.
   - Porém, no **teste retido (`test`)**, a mesma remoção reduziu `mAP50-95` (`0,5051 -> 0,4038`) e `freight_car mAP50` (`0,4645 -> 0,3842`), elevando os falsos negativos de `226` para `286`.
   - **Causa raiz medida:** o dataset `TWIST detailed_labels` possui apenas **`5` eventos de carga** no total (`3` treino / `1` val / `1` test). Como o evento `15` (`test`) apresenta vagões rápidos sob luz diurna intensa e forte *motion blur*, enquanto `exp02` perdeu metade dos frames de `high_speed_train` (`187 -> 92`), `60` amostras de `freight_car` e mais de metade dos fundos negativos (`328 -> 150`), a poda cega por `dHash <= 2` privou o modelo de amostras de transição/borramento que sustentavam a robustez no evento de teste — confirmando exatamente por que near-duplicates não devem ser descartados automaticamente sem medir o impacto.
2. **Comparação `YOLO26n` vs. `YOLO26s` (`exp02` vs. `exp03`):**
   - Justificada pela margem disponível em `val` (`recall=0,6549`, `mAP50=0,6720 <= 0,85`), a comparação controlada mostrou que **`YOLO26s` não superou `YOLO26n` na validação** (`val mAP50-95 = 0,4797` vs. `0,5031`; `val mAP50 = 0,6528` vs. `0,6720`), sofreu *early stopping* na época `36`, colapsou o *recall* de `passenger_wagon` no teste (`0,053`) e consumiu `~1,8x` mais VRAM (`4572,8 MiB`) com latência `~2,17x` maior (`5,263 ms` vs. `2,420 ms` por imagem em `val`).
   - Embora no `test` o `YOLO26s` tenha obtido `freight_car mAP50=0,5978` e `locomotive mAP50=0,8931` superiores ao `exp02`, **a seleção seguiu estritamente o protocolo sem vazamento (`val mAP50-95`), mantendo `exp02` como checkpoint PoC para rastreamento**.
3. **Análise qualitativa dos erros no *test set* (`exp01/02/03_test_errors.jpg`):**
   - Nos trens de alta velocidade (`ICE`), a anotação original do `TWIST` sobrepõe uma caixa externa `high_speed_train` e múltiplas caixas internas pequenas de `passenger_wagon` indo em direção ao ponto de fuga; os modelos tendem a detectar a envoltória global (`high_speed_train mAP50` entre `0,954` e `0,985` no teste) e omitir os carros distantes internos.
   - Nos trens de carga (`train_2023-12-11_08.36.55_029.jpg` e `08.36.56_557.jpg`), `YOLO26n` acerta `1–2` vagões próximos em primeiro plano, mas falha nos vagões distantes e em uma locomotiva parcialmente recortada na borda esquerda (`exp01`), enquanto `exp03` para completamente de emitir caixas em frames com *blur* mais forte.

---

## 3. Avaliação de Rastreamento e Vídeos Públicos Não Vistos (`ml/scripts/eval_video_tracking.py`)

> [!NOTE]
> **Confirmação de CUDA e significado métrico:**
> - **Dispositivo utilizado:** todas as execuções de tracking (`vid_09`) e detecção pontual (`vid_05`, `vid_04`) foram processadas em **`cuda:0` (`NVIDIA GeForce RTX 4060 Ti`)** via `device=0` (`torch.cuda.is_available() == True`), confirmada pelo uso medido de VRAM (`60,0 MiB` ByteTrack, `64,4 MiB` BoT-SORT, `55,3 MiB` COCO Ref) e verificada em sonda pós-execução (`predictor.device=cuda:0`, parâmetros em `cuda:0`, latência estabilizada de inferência `track()` em `19,54 ms/frame`). O script `eval_video_tracking.py` foi adicionalmente atualizado para registrar `"execution_environment"` e `"device"` explicitamente em todas as execuções futuras (sem sobrescrever o JSON da Fase 3).
> - **Cobertura de detecção ≠ recall/acurácia:** como `vid_09`, `vid_05` e `vid_04` **não possuem caixas *ground truth* anotadas na Fase 3**, percentuais de frames com ao menos uma detecção (`24,4%`, `25,0%`, `100%`, `1/6`, `2/6`) representam exclusivamente **cobertura de detecção**, e **não** *recall*, *precision* ou acurácia.

Resultados medidos com `exp02/weights/best.pt` (`conf=0,25`, `imgsz=640`, `cuda:0`) comparados à referência pré-treinada COCO (`yolo26n.pt`, classe `6 = train`):

### 3.1 `vid_09` (`Vossloh G 1206 BB` — PoC com ~25 vagões *hopper* em movimento, `1.245` frames)

| Execução | Tracker | Cobertura de detecção (`frames c/ det / total`, **sem GT**) | Total detecções brutas | `track_id` únicos | Curtas (`<10` fr.) | Sustentadas (`>=25` fr.) | Gaps (`>3` fr.) | Mediana / Máx. frames | Velocidade (`fps`) / VRAM (`cuda:0`) | Classes dominantes preditas |
|---|---|---|---|---|---|---|---|---|---|---|
| **TWIST `exp02`** | `bytetrack.yaml` | `304 / 1.245` (`24,4%` cobertura) | `305` | `15` | `6` | `6` | `7` | `14 / 48` | `36,12 fps` (`34,47s`) / `60,0 MiB` | `14` faixas `high_speed_train` + `1` `locomotive` (`track_id=40`, frames `1236..1244`) |
| **TWIST `exp02`** | `botsort.yaml` | `311 / 1.245` (`25,0%` cobertura) | `313` | `15` | `6` | `6` | `8` | `14 / 48` | `23,42 fps` (`53,15s`) / `64,4 MiB` | `14` faixas `high_speed_train` + `1` `locomotive` (`track_id=32`, frames `1236..1244`) |
| **COCO Ref (`class=6`)** | `bytetrack.yaml` | `1.245 / 1.245` (`100%` cobertura) | `1.245` | `1` (`track_id=1`) | `0` | `1` | `0` | `1.245 / 1.245` | `34,87 fps` (`35,71s`) / `55,3 MiB` | `100% train` (caixa única para o comboio inteiro) |

- **Inspeção visual (`vid09_twist_bytetrack_samples.jpg`, `vid09_twist_botsort_samples.jpg`, `vid09_coco_ref_bytetrack_samples.jpg`):**
  - Nas passagens próximas dos *hoppers* marrons com brita (`t=0s` a `t=39,8s`), o modelo treinado apenas no `TWIST` não consegue isolar vagões *hopper* individuais em ângulo oblíquo elevado: ou não emite detecções (`conf < 0,25`), ou rotula segmentos de múltiplos vagões como `high_speed_train`, identificando corretamente apenas a locomotiva vermelha/branca de cauda (`locomotive 0,43` em `f=1244`).
  - `BoT-SORT` recuperou ligeiramente mais continuidade em duas trilhas (`13` vs. `10` frames em `track_id=7`; `30` vs. `26` em `track_id=28`), mas com `~35%` menor vazão (`23,42 fps` vs. `36,12 fps` no `ByteTrack`). Contudo, nenhum rastreador compensa a falha do detector em separar vagões individuais sob mudança severa de ângulo.
  - A referência COCO (`1.245/1.245` frames de cobertura) comprova que o pré-treino generalista já reconhece o **trem como objeto macro** de cima (`0,55` a `0,91`), porém funde todos os ~25 vagões em uma única *bounding box* (`track_id=1`), inviabilizando contagem individual ou inspeção por vagão sem anotações dedicadas de vagões abertos em perspectiva elevada.

### 3.2 `vid_05` (*Timelapse* — apenas análise visual e detecção pontual, SEM tracking contínuo)
- Frames inspecionados (`[10, 66, 140, 142, 260, 269]`, `vid05_twist_point_detection.jpg` vs. `vid05_coco_ref_point_detection.jpg`):
  - **TWIST `exp02`:** cobertura em `1/6` frames (`f=140`: falso rótulo `high_speed_train 0,63` sobre vagões-tanque brancos); falhou nos *hoppers* marrons visíveis em `f=142`, nas locomotivas em `f=10` e `f=260` e nos vagões fechados em `f=66` e `f=269`.
  - **COCO Ref (`train`):** cobertura em `6/6` frames (`train 0,88` a `0,92`), novamente apenas no nível de composição inteira.

### 3.3 `vid_04` (Passarela elevada / vagões estacionados na neve, 6 frames `[0, 340, 645, 725, 806, 847]`)
- Inspeção visual (`vid04_twist_overpass_detection.jpg`):
  - **TWIST `exp02`:** cobertura em `2/6` frames sem GT de caixas (`0` detecções nos frames `0, 340, 725, 806`; `1` falso positivo `passenger_wagon` em `f=645`; e em `f=847` uma caixa `locomotive` mal localizada englobando o pórtico metálico de catenária e vagões-tanque estacionados no pátio, sem isolar os vagões abertos verdes estacionados na via paralela).

---

## 4. Limitações Documentadas e Próximo Passo

1. **Ausência de generalização para vagões abertos:** os resultados da Fase 3 **não demonstram generalização para vagões abertos (`open_top_freight_wagon`)** em ângulo oblíquo elevado.
2. **Câmera única no treinamento (`TWIST`):** `TWIST detailed_labels` contém apenas visão lateral/baixa em clima de inverno. A transferência para câmeras oblíquas elevadas (`vid_09`, `vid_05`, `vid_04`) falhou no nível de vagão individual.
3. **Amostra reduzida de eventos de carga no TWIST:** Apenas `5` eventos de carga existem no `detailed_labels` (`val` e `test` avaliam `freight_car` sobre `1` único trem cada), gerando alta variância entre splits e divergência acentuada entre `conf ≈ 0,03` e `conf = 0,25`.
4. **Semântica de classes limitada:** O detector atual não distingue vagão aberto de fechado/tanque, nem identifica material ou nível de carga (**Fase 5** mantida **pendente** até obtenção de mais fontes abertas diversas — ver `specs/dataset-audit.md`).
5. **Etapa intermediária imediata (Fase 3.1) antes do Spike Java/ONNX (Fase 4):** conforme aprovado, será preparada uma amostra manual especializada focada na classe **`open_top_freight_wagon`** sobre os vídeos `vid_09`, `vid_14`, `vid_15`, `vid_16` e `vid_04`, separada por campanha/fonte e com um trecho sequencial anotado com identidades para benchmark real de `ByteTrack` vs. `BoT-SORT` (ver proposta em `specs/experiments/phase3_1-open-top-sampling-plan.md`).

## Artefatos locais preservados (fora do Git)
- Pesos Fase 3 preservados para comparação futura: `artifacts/runs/detect/exp01_yolo26n_twist_exactdedup/weights/best.pt`, `artifacts/runs/detect/exp02_yolo26n_twist_neardedup_ablation/weights/best.pt`, `artifacts/runs/detect/exp03_yolo26s_twist/weights/best.pt`
- Resumos JSON preservados: `artifacts/runs/detect/exp01..03/eval_summary.json` e `artifacts/runs/tracking/phase3_video_tracking_eval.json`
- Pranchas visuais inspecionadas: `data/processed/triage-frames/exp01..03*_test_errors.jpg`, `vid09_twist_bytetrack_samples.jpg`, `vid09_twist_botsort_samples.jpg`, `vid09_coco_ref_bytetrack_samples.jpg`, `vid05_twist_point_detection.jpg`, `vid05_coco_ref_point_detection.jpg`, `vid04_twist_overpass_detection.jpg`
