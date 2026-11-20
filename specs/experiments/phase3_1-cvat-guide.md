# Guia Operacional no CVAT — Fase 3.1 (`open_top_freight_wagon`)

> [!IMPORTANT]
> **Regra metodológica antes de anotar:**
> 1. **Não inicie treinamento** antes de concluir o *smoke test*, as duas tarefas e a revisão final das caixas/IDs.
> 2. **Não use frames do clipe de tracking de teste no treinamento.**
> 3. **Métricas MOT sobre 5 fps:** no vídeo/sequência da Tarefa 2 (`vid_09`, `t=12,0s..24,0s`), o tracking será avaliado **estritamente sobre os 61 frames amostrados a `5 fps` (`frames nativos 300..600`, passo 5)** — e não sobre os 301 frames nativos de `25 fps`.

---

## 1. Pacotes e Arquivos Gerados Localmente (fora do Git)

Pasta principal: `data/processed/open-top-v1/cvat_tasks/`  
Manifesto rastreado de splits e frames: [open-top-split-v1.json](../data-manifests/open-top-split-v1.json)

- **Definição de classe única para o Projeto no CVAT:** `cvat_labels_open_top.json` (`0: open_top_freight_wagon`, tipo `rectangle`).
- **Pacotes de *Smoke Test* (Condição 1):**
  - Detecção (`5` imagens): `smoke_test_detection_5imgs.zip`
  - Tracking (`5` frames, `5 fps`): `smoke_test_tracking_5frames.zip` *(preferencial pela ordem exata 1:1)* ou `smoke_test_tracking_5frames.mp4`
- **Pacotes Completos de Anotação:**
  - **Tarefa 1 (Detecção — `141` imagens):** `task1_detection_141imgs.zip` (arquivos nomeados como `<split>__<campanha>__<video>__f<frame_nativo>.jpg`)
  - **Tarefa 2 (Tracking GT `vid_09` — `61` frames @ `5 fps`):** `task2_tracking_vid09_5fps_61frames.zip` *(recomendado: sequência numerada `000001.jpg`..`000061.jpg`)*; opção equivalente em vídeo: `vid09_tracking_5fps_61frames.mp4` (ambos exatamente com `61` quadros).

---

## 2. Passo 1 — Smoke Test de Importação e Exportação (Condição 1)

Crie um **Projeto** no CVAT chamado `railvision_open_top_v1` e cole/importe o conteúdo de `cvat_labels_open_top.json` (garantindo apenas a classe `open_top_freight_wagon` de índice `0`).

### 2.1 Smoke Test de Detecção (`5` imagens)
1. Dentro do projeto, crie uma tarefa (`smoke_det`) importando **`smoke_test_detection_5imgs.zip`** (Sorting method: **lexicographical**, Image quality `100`).
2. Anote pelo menos `1` imagem com caixa `open_top_freight_wagon` e deixe intencionalmente a imagem `test__camp_vfli_cosne__vid_16__f000050.jpg` (só locomotivas) vazia (`0` caixas) para testar negativos sem erro.
3. Exporte em **`Actions -> Export task dataset -> Ultralytics YOLO Detection 1.0`** (marque *Save images* opcionalmente; descompacte em uma pasta temporária).
4. Execute no terminal do projeto:
```powershell
ml\.venv\Scripts\python.exe ml\scripts\validate_cvat_exports.py --mode smoke-detection --export-dir C:\caminho\para\smoke_det_yolo
```
5. **Critério de aprovação:** saída JSON sem erros, `expected_images: 5`, `missing_label_files: []` (ou apenas o negativo caso o formato não gere `.txt` vazio) e classe única `0`.

### 2.2 Smoke Test de Tracking (`5` frames)
1. Crie uma tarefa (`smoke_trk`) importando **`smoke_test_tracking_5frames.zip`** (preferível por garantir frames `1..5` exatos no CVAT sem reamostragem de codec).
2. No quadro `1` (`000001.jpg`), crie **Track** (não *Shape*) de `open_top_freight_wagon` em 1 ou 2 vagões, avance para o quadro `5` (`000005.jpg`) e ajuste a caixa para ativar a interpolação automática entre `1` e `5`.
3. Exporte em **`Actions -> Export task dataset -> MOT 1.1`**, descompacte e aponte para `gt/gt.txt`:
```powershell
ml\.venv\Scripts\python.exe ml\scripts\validate_cvat_exports.py --mode smoke-tracking --mot-file C:\caminho\para\smoke_trk_mot\gt\gt.txt
```
4. **Critério de aprovação:** `expected_num_frames: 5`, `frames_with_annotations: 5`, `unique_track_ids` consistentes e sem erros de geometria ou duplicidade `(frame_id, track_id)`.

---

## 3. Passo 2 — Anotação Completa no CVAT

### 3.1 Tarefa 1 — Detecção (`141` imagens)
1. Crie a tarefa `task1_open_top_detection_141imgs` no mesmo projeto importando **`task1_detection_141imgs.zip`** (**Sorting method: lexicographical**).
   - *Observação:* Não separe em subconjuntos internos aleatórios no CVAT — o prefixo do nome do arquivo (`train__...`, `val__...`, `test__...`) já trava a separação estrita por campanha.
2. **Regras de anotação (`open_top_freight_wagon`):**
   - Use **Shape (Rectangle)** para anotar o corpo externo completo de cada vagão aberto de carga a granel.
   - **Exclua:** locomotivas, tanques cilíndricos, vagões fechados (`vid_04` trem central), contêineres e pórticos.
   - **Corte de ponto de fuga/oclusão:** anote apenas vagões com **lado menor `>= 24 px`** (`>= 18 px` em `vid_04` `480p`), **`>= 40%` visível** e **fronteira entre vagões distinguível** a olho nu. Não desenhe caixa única sobre vagões fundidos ao longe.
   - Em quadros só de locomotivas (*hard negatives* em `vid_14` e `vid_16` `t=2..10,5s`), deixe `0` caixas se nenhum vagão aberto elegível estiver presente.
3. Exporte em **`Ultralytics YOLO Detection 1.0`** e descompacte em `data/processed/open-top-v1/exports/yolo_det/`.

### 3.2 Tarefa 2 — Tracking GT `vid_09` (`61` frames a `5 fps`)
1. Crie a tarefa `task2_vid09_tracking_5fps_61frames` importando **`task2_tracking_vid09_5fps_61frames.zip`** (**Sorting method: natural/lexicographical**; os nomes `000001.jpg`..`000061.jpg` garantem correspondência exata quadro a quadro).
2. Use sempre o modo **`Track`** (nunca *Shape*) para cada vagão elegível pela regra de corte (`>= 24 px`, `>= 40%` visível):
   - Coloque *keyframes* nos extremos e nos pontos médios ao passar atrás do arbusto à direita ou postes de catenária; deixe o CVAT interpolar os quadros intermediários.
   - Se o CVAT trocar IDs ao sair/entrar ou após edição, use **`Merge`** / **`Split`** para preservar a mesma trajetória do vagão enquanto ele permanecer visível nos `61` quadros.
3. Exporte em **`MOT 1.1`** e extraia o arquivo `gt/gt.txt` para `data/processed/open-top-v1/exports/mot_vid09/gt/gt.txt`.

---

## 4. Passo 3 — Validação, Ingestão Anti-Vazamento e Revisão (Condição 3)

Após exportar e descompactar ambas as tarefas, execute o script de validação e ingestão antes de qualquer treinamento:

```powershell
ml\.venv\Scripts\python.exe ml\scripts\validate_cvat_exports.py `
  --mode ingest-full `
  --det-export-dir data\processed\open-top-v1\exports\yolo_det `
  --mot-file data\processed\open-top-v1\exports\mot_vid09\gt\gt.txt
```

O script irá:
1. Validar classe `0`, coordenadas normalizadas e ausência de vazamento entre campanhas (bloqueando qualquer quadro de tracking em `train` ou `val`).
2. Organizar `data/processed/open-top-yolo-v1/{train,val,test}/{images,labels}` + `data.yaml` + `tracking_gt/vid09_5fps_gt.txt`.
3. Gerar 4 pranchas visuais para revisão humana (**Condição 3**) em `data/processed/triage-frames/`:
   - `qa_detection_train.jpg`, `qa_detection_val.jpg`, `qa_detection_test.jpg`
   - `qa_tracking_vid09_5fps.jpg`

Somente após revisar essas pranchas e aprovar eventuais ajustes no CVAT avançaremos para o *fine-tuning* (`YOLO26n` em `cuda:0`) e o benchmark `ByteTrack` vs. `BoT-SORT` sobre os `61` frames.
