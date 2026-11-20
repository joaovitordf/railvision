# Model Card — `yolo26n-twist-poc` (Fase 3 / Camada C+)

## 1. Identificação do Modelo
- **Nome lógico do experimento:** `exp02_yolo26n_twist_neardedup_ablation` (preservado como baseline pré-especialização da Fase 3)
- **Arquitetura base:** `YOLO26n` (`Ultralytics 8.4.215`, `PyTorch 2.9.1+cu128`, entrada `640x640`)
- **Artefato local (fora do Git):** `artifacts/runs/detect/exp02_yolo26n_twist_neardedup_ablation/weights/best.pt`
- **Status:** **Prova de Conceito (PoC) — NÃO apto para produção e NÃO generaliza para vagões abertos (`open_top_freight_wagon`).**

## 2. Uso Pretendido e Usos Fora de Escopo
- **Uso pretendido:** Referência de baseline pré-especialização da Fase 3 e comparação contra o futuro *fine-tuning* da **Fase 3.1** (`open_top_freight_wagon`).
- **Usos estritamente proibidos / fora de escopo:**
  - Não utilizar em operação ferroviária real ou tomada de decisão logística/segurança.
  - Não usar para detectar ou distinguir **`open_top_freight_wagon`** / **`freight_wagon_open`** (o rótulo `freight_car` do TWIST engloba pranchas, tanques, baús e gôndolas mistos, e falhou severamente em ângulo oblíquo elevado).
  - Não usar como estimador de **material transportado** nem **nível de carregamento**.
  - Não assumir desempenho válido em câmeras aéreas/oblíquas elevadas sem *fine-tuning* especializado multi-campanha.

## 3. Dados de Treinamento e Licenciamento
- **Dataset:** `TWIST (Tracks, Wagons, Inspection, Signs, Trains)` — subconjunto `detailed_labels` (`train-only near-dedup`, `698` imagens de treino em `31` eventos distintos; `val = 265` em `8` eventos; `test = 161` em `7` eventos).
- **Licença de origem:** Conforme `specs/licenses.md`, revisar restrições do `TWIST` antes de qualquer redistribuição pública de pesos derivados.
- **Classes de saída (`4`):** `0: locomotive`, `1: passenger_wagon`, `2: freight_car`, `3: high_speed_train`.

## 4. Métricas Quantitativas Medidas (`NVIDIA GeForce RTX 4060 Ti`, `cuda:0`)
- **Conjunto de Validação (`val`, 265 imgs, 8 eventos · `model.val()` no ponto ótimo de `F1`):**
  - `Precision = 0,6717` · `Recall = 0,6549` · `mAP50 = 0,6720` · `mAP50-95 = 0,5031`
  - Por classe (`mAP50 / mAP50-95`): `locomotive = 0,2233 / 0,1771`, `passenger_wagon = 0,6267 / 0,3911`, `freight_car = 0,9337 / 0,6878` (`P = 0,8901, R = 0,8806`), `high_speed_train = 0,9041 / 0,7562`.
  - Latência de inferência em `val`: `2,420 ms/img` (+ `0,710 ms` pré / `1,904 ms` pós); VRAM pico de treino: `2541,9 MiB`.
- **Conjunto Retido de Teste (`test`, 161 imgs, 7 eventos, `361` caixas GT):**
  - **`model.val()` (média macro no limiar ótimo de `F1` médio ≈ `0,0300`):** `Precision = 0,7827` · `Recall = 0,4697` · `mAP50 = 0,5679` · `mAP50-95 = 0,4038` (`freight_car = 0,3842 / 0,2907`, `locomotive = 0,6051 / 0,4186`, `passenger_wagon = 0,3097 / 0,1301`, `high_speed_train = 0,9725 / 0,7757`).
  - **Avaliação operacional fixa (`model.predict`, `conf >= 0,25`, `IoU >= 0,50`):** `TP = 75`, `FP = 19`, `FN = 286` → **`micro_P = 0,7979`**, **`micro_R = 0,2078`** (`macro_P = 0,8989`, `macro_R = 0,2416`). A diferença em relação a `0,7827 / 0,4697` decorre do corte mais rigoroso em `conf=0,25`, da agregação micro vs. macro e do *pipeline* discreto vs. curva suavizada.
- **Execução *cross-domain* em vídeos públicos não vistos (`cuda:0`, **sem GT — apenas cobertura de detecção**):**
  - `vid_09` (*hopper* em movimento, ângulo oblíquo elevado): cobertura de frames com alguma detecção de **`24,4%`** (`ByteTrack`, `36,12 fps`, `60,0 MiB` VRAM) / **`25,0%`** (`BoT-SORT`, `23,42 fps`, `64,4 MiB` VRAM) — **isso é cobertura bruta de detecção, NÃO recall nem acurácia**; visualmente houve erro semântico (`high_speed_train` sobre *hoppers*) e acerto localizado apenas na locomotiva traseira (`0,43`).

## 5. Riscos, Vieses e Limitações
- **Sem generalização para vagões abertos:** não detecta vagões abertos individuais de forma utilizável sob ângulo oblíquo elevado.
- **Viés severo de câmera única e estação:** Câmera fixa rente ao solo no inverno europeu.
- **Variância estatística alta em carga:** Apenas `1` evento de carga em `val` (`67` instâncias) e `1` em `test` (`107` instâncias).
- **Queda sob *motion blur*, mudança de ângulo e elevação de limiar:** Confirmada no `test` do TWIST em `conf=0,25` e em `vid_09`, `vid_05` e `vid_04`.
