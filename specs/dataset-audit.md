# RailVision AI — Auditoria de Datasets e Triagem de Vídeos Públicos (Portão P1)

**Documento fonte:** Especificação Unificada v1.2  
**Data da auditoria medida:** 2026-11-19 / 2026-11-20  
**Manifestos versionados:** [datasets-manifest.json](data-manifests/datasets-manifest.json) · [public-videos-manifest.json](data-manifests/public-videos-manifest.json)

> **Regra de veracidade:** Todos os números abaixo na Seção 2 foram medidos diretamente nos arquivos baixados ou via índice ZIP64 remoto / OpenCV e inspecionados visualmente em grades de frames. Nenhuma métrica de modelo foi inventada.

---

## 1. Resumo dos Datasets Públicos Candidatos

### 1.1 TWIST (*Trains under Weather, Illumination, and Seasonal Time*)
- **Fonte:** <https://zenodo.org/records/19472084> | Repositório: <https://github.com/ds-kiel/TWIST>
- **Status:** **Baixado, extraído e auditado** (`labelled_dataset.tar.gz`, `596.144.002` bytes, `MD5=64d50175ce51c52f292c688a37c7a453` verificado, `SHA-256=b6194915dd95070fa1afa2a99c8f44a03e6877fd5068030784cf655d86debda2`). *(Arquivo `unlabelled_data.tar.gz` de 1,36 GB não foi baixado nesta triagem).*
- **Licença verificada:** `CC-BY-4.0` (exige atribuição).
- **Contagens medidas no arquivo `labelled_dataset.tar.gz`:**
  - Total: **11.492** imagens (`100%` em resolução `640x480`, `0` corrompidas) e **11.492** arquivos `.txt`.
  - **Duplicatas:** **812** duplicatas exatas (SHA-256) e **4.879** correspondências próximas (*dHash* distância ≤ 2 em janela deslizante) no arquivo completo; especificamente em `detailed_labels/` (`1.492` imagens), foram medidos **5 grupos de duplicatas exatas (8 arquivos extras)**, **46 eventos distintos de passagem de trem** (separados por intervalo > 120s entre `2023-12-08` e `2024-03-16`, sendo **5 eventos** contendo `freight car`) e **2 linhas de bounding box com área zero** (`w=0, h=0`).
  - **`binary_labels/` (10.000 imagens):** `train` = 8.000, `validation` = 1.000, `test` = 1.000; `1.884` arquivos vazios (negativos sem trem); `8.139` caixas positivas todas com `class_id = 6` (`train` no COCO); `1` linha inválida.
  - **`detailed_labels/` (1.492 imagens — e não 1.493 como descrito no registro Zenodo):** `0` vazios, `2` linhas inválidas. Mapeamento oficial encontrado em `detailed_labels/class_mapping.md`:
    - `0` (`train`): **1.472** caixas
    - `1` (`locomotive`): **165** caixas
    - `2` (`wagon`): **1.085** caixas
    - `3` (`freight car`): **747** caixas
    - `4` (`ICE`): **295** caixas
    - `5` (`other`): **0** caixas
    - `6` (`object`): **1.503** caixas
- **Achados críticos na inspeção visual (`twist_freight_car_vs_wagon.jpg`):**
  1. **`wagon` (ID 2) vs. `freight car` (ID 3):** `wagon` rotula **carros de passageiros** (inclusive trens regionais de dois andares), enquanto `freight car` rotula **vagões de carga** (tanques, contêineres, fechados). **Nenhuma dessas duas classes do TWIST equivale automaticamente a vagões abertos (*open-top*).**
  2. **Colisão de ID `6`:** em `binary_labels/`, o ID `6` representa o **trem inteiro**; em `detailed_labels/`, o ID `6` (`object`) rotula **objetos fixos na frente da câmera** (placa amarela de travessia de pedestres e gradil metálico). **Nunca mesclar as duas pastas sem remapear IDs.**
  3. **Perspectiva e domínio:** câmera fixa única ao nível da via, com chuva/neve/noite/blur acentuado; **não mostra o interior nem a superfície da carga de vagões abertos**.

### 1.2 UFOP Iron Ore Conveyor Dataset
- **Fonte:** <https://data.mendeley.com/datasets/s25x2bnshz/1> | **Licença:** `CC BY 4.0`
- **Status:** **Índice ZIP64 remoto inspecionado via HTTP Range** (`image-data.zip`, `6.228.464.374` bytes, `SHA-256` declarado `f4ced924f274bcd729fd77ab0aca8fa72a48b5419077e2862b87ab175fd12480`) sem baixar 5,8 GiB.
- **Contagens medidas no índice central ZIP64:** `69.636` arquivos (`42.744 .jpeg`, `24.398 .jpg`, `2.474 .txt`, `19 .mp4`, `1 data.yaml`).
- **Conteúdo real (`yolo-dataset-center/data.yaml`):** `nc: 1`, `names: ['conveyor']`. As pastas dividem amostras laboratoriais de esteira com minério (`normal`) vs. objetos estranhos/madeira (`annomaly`). **Não possui rótulos multi-material ferroviário nem de vagões.** Uso restrito a referência visual fora do domínio alvo.

### 1.3 Wagons Images Classification (Hugging Face)
- **Licença:** `CC-BY-NC-ND-4.0`, `18` exemplos (`loaded`/`unloaded`). **Excluído de treinamento** por licença e volume insuficiente (Q19).

---

## 2. Triagem de Vídeos Públicos Licenciados (Wikimedia Commons)

Critério aplicado em `audit_policies.py`: aceitos apenas vídeos com licença explícita `CC0`, `CC BY` ou `CC BY-SA` (`25` candidatos aprovados pela licença dentre `48` retornados nas buscas, `23` rejeitados por licença/tipo/tamanho). Foram baixados e inspecionados visualmente **9 vídeos**:

| ID | Resolução / FPS / Duração | Licença | Categoria de ângulo observada | Carga visível? | Achado na inspeção visual |
|---|---|---|---|---|---|
| `vid_09_vossloh_g_1206_bb_of_vfli_video` | `1920x1080` · `25 fps` · `49.80s` (`135,9 MB`) | `CC BY-SA 4.0` | `elevated_oblique_cargo_visible` | **Sim** | **Melhor vídeo encontrado:** ~25 vagões *hopper* abertos carregando brita/lastro cinza-claro (`limestone`/agregado aparente; `label_basis=source_metadata+visual_only`), visão oblíqua elevada limpa do interior. |
| `vid_05_timelapse_freight_bypass_railway_eichenb` | `1920x1080` · `25 fps` · `13.52s` (`19,9 MB`) | `CC BY-SA 4.0` | `elevated_oblique_cargo_visible` | **Parcial** | Câmera fixa em ponte rodoviária sobre a via; porém é **timelapse** (8 trens em 13,5s, ~2–5 frames por trem). Em `t=5.7s` aparecem vagões *hopper* abertos com carga escura/marrom-avermelhada (`visual_only`) e vagões com bobinas. |
| `vid_04_2es10_003_with_freight_train_from_above` | `640x480` · `15 fps` · `56.73s` (`7,1 MB`) | `CC BY 3.0` | `elevated_oblique_cargo_visible` | **Parcial (estacionados)** | Passarela elevada na neve; o trem central em movimento tem locomotiva + vagões fechados; na via paralela há vagões abertos verdes e tanques parcialmente visíveis em ângulo oblíquo. |
| `vid_06_seattle_freight_train_moves_across_the_s` | `1920x1080` · `29.97 fps` · `15.42s` (`21,5 MB`) | `CC BY-SA 4.0` | `side_or_ground_level_cargo_not_visible` | Não | Vista lateral por ponte metálica com trens de contêineres *double-stack* (alta oclusão pelas treliças). |
| `vid_07_freight_train_crossing_the_kyu_omono_riv` | `1920x1080` · `24 fps` · `67.08s` (`37,8 MB`) | `CC BY-SA 4.0` | `side_or_ground_level_cargo_not_visible` | Não | **Excluído do treino de vagões abertos:** vista lateral em ponte fluvial com vagões-prancha de contêiner (incompatível com perspectiva de carga a granel *open-top*). |
| `vid_08_dampfloks_auf_der_rampe_tharandt_klingen` | `720x576` · `~25 fps` efetivo (`74.33s`, `1.858` frames) (`23,0 MB`) | `CC BY-SA 4.0` | `side_or_ground_level_cargo_not_visible` | Não | Locomotivas a vapor históricas em curva; muita fumaça e texto sobreposto inicial. **Nota técnica:** OpenCV reportou `fps=30000.0` por *timebase* WebM VFR — caso de teste importante para a Fase 4. |
| `vid_01_langer_unbeladener_g_terzug_von_oben_k_l` | `3840x2160` · `29.97 fps` · `116.45s` (`54,4 MB`) | `CC BY-SA 4.0` | `other_or_unusable` | Não | Drone muito distante entre árvores; vagões-prancha de contêiner vazios pequenos demais para inspeção de carga. |
| `vid_02_soundscapes_of_norrk_ping_9_bridge` | `1920x1080` · `30 fps` · `56.00s` (`50,0 MB`) | `CC BY 3.0` | `other_or_unusable` | Não | Falso positivo de busca (ponte rodoviária com carros). |
| `vid_03_stadler_flirt_emu_1323_and_1311_at_vesse` | `1280x720` · `30 fps` · `172.63s` (`74,5 MB`) | `CC BY 3.0` | `other_or_unusable` | Não | Trem elétrico de passageiros (EMU) em primeiro plano; pátio de carga muito distante ao fundo. |
| `vid_10`–`vid_13` *(busca paralela)* | `480x268` a `1920x1080` | `CC BY 3.0` / `CC BY-SA 3.0/4.0` | `other_or_unusable` / `side_or_ground_level` | Não | Falsos positivos da palavra *ballast* ou equipamentos de manutenção de via sem carga a granel (`vid_10` lâmpada, `vid_11` noturno, `vid_12` socadora, `vid_13` oclusão lateral). |
| `vid_14_coal_train_locomotive_exit_the_tunnel_cr` | `1280x720` · `59.94 fps` · `1384.88s` (`358,2 MB`) | `CC BY 3.0` | `elevated_oblique_cargo_visible` | **Sim (trechos)** | **Descoberta paralela para futura Fase 5:** vídeo de 23 min com trechos em talude elevado (`t=553.9s`, `t=830.9s`) mostrando topo/bordas de vagões abertos carregados com **carvão** (`label_basis=source_metadata+visual_only`). |
| `vid_15_sishen_saldanha_railway_line` | `1920x1080` · `29.97 fps` · `26.36s` (`12,5 MB`) | `CC BY-SA 3.0` | `elevated_oblique_cargo_visible` | **Não (paredes altas)** | Trem pesado de minério de ferro Sishen–Saldanha em ângulo oblíquo elevado; paredes altas das gôndolas ocluem quase toda a superfície do minério (útil para geometria de vagões abertos, não para superfície de material). |
| `vid_16_vossloh_g_1206_bb_61735_and_bb_61744_vfl` | `1920x1080` · `25 fps` · `20.40s` (`51,4 MB`) | `CC BY-SA 4.0` | `elevated_oblique_cargo_visible` | **Sim** | **Descoberta paralela para futura Fase 5:** passarela sobre via férrea (`Cosne-sur-Loire`, mesmo autor `Cjp24` de `vid_09`); em `t=16.3s..20.4s` mostra o topo aberto do primeiro vagão *hopper* com lastro/brita cinza-clara (`transport de ballast neuf`). Mesma localidade/operadora de `vid_09` (não conta como local independente). |

#### Outras candidatas avaliadas em triagem leve (rejeitadas sem download integral)
- **Hugging Face `UniqueData/wagons-images-classification`** *(inspecionado via API Hugging Face)*: rejeitado por licença **`cc-by-nc-nd-4.0`** (cláusulas NC e ND incompatíveis com extensão/derivação) e volume irrisório (**18 exemplos** no split `train`).
- **Wikimedia `File:Отцеп вагонов v001.webm`** *(`CC BY-SA 4.0`, `10,51 MB`)*: rejeitado na triagem visual da miniatura (`marshaling_yard_thumb.jpg`) — visão lateral rente ao solo de gôndolas verdes com paredes altas que ocluem completamente o interior/superfície da carga.
- **Wikimedia buscas específicas (`bridge`, `aerial`, `drone`, `limestone`, `sand`, `gravel`)**: retornaram principalmente locomotivas/trens de passageiros, contêineres intermodais (`vid_06` já auditado) ou falsos positivos não ferroviários (`Alvord Desert`, `Deschutes River`); **nenhuma fonte aberta adicional independente para `iron_ore`, `limestone` (além da brita de `vid_09`/`vid_16`), ou `sand`** com superfície claramente visível foi localizada até o momento, mantendo a **Fase 5 pendente**.

---

## 3. Conclusão e Recomendação para o Portão P1

### Diagnóstico objetivo
1. **Detecção e Rastreamento de Vagões (Viável com ressalvas de domínio):** TWIST fornece `747` caixas de `freight car` (`640x480`), porém todas de uma câmera baixa fixa e com alta redundância temporal. Para evitar sobreajuste, será necessário deduplicar o TWIST, remapear o ID `6` corretamente e **combinar quadros extraídos dos vídeos públicos elevados (`vid_09`, `vid_05`, `vid_04`) e laterais (`vid_07`)**.
2. **Material e 5 Níveis Ordinais de Carregamento (Camada A NÃO sustentada pelas fontes atuais):**
   - Apenas **`vid_09`** (1 único trem/sessão com ~25 *hoppers* de brita/lastro) e poucos frames de *timelapse* em **`vid_05`** mostram claramente a superfície de carga a granel de ângulo elevado oblíquo.
   - Não há nas fontes públicas abertas auditadas até aqui múltiplos vídeos/sessões independentes para **`iron_ore`**, **`coal`**, **`sand`** ou variação equilibrada entre `empty / low / medium / high / very_high`.
   - Treinar 4–5 classes de material ou 5 níveis ordinais sobre 1 única sessão violaria a regra de admissão de classes e a política de *split* por grupo (vazamento entre treino e teste).

### Decisão Aprovada no Portão P1 (2026-11-20)
- **Escopo aprovado para Fase 3:** **Camada C+ (Prova de Conceito de Detecção e Rastreamento de Vagões)**.
  1. Baseline inicial **`YOLO26n`** no TWIST `detailed_labels` (split estrito por **evento de passagem**, zero vazamento; removendo apenas as `8` duplicatas exatas e avaliando em ablação o impacto de filtrar correspondências próximas), usando apenas classes compatíveis (`locomotive`, `passenger_wagon`, `freight_car`, `high_speed_train` — sem equiparar `freight_car` a `freight_wagon_open` e excluindo `train`/`object`). Comparação com `YOLO26s` somente se os resultados justificarem.
  2. Avaliação de **`ByteTrack`** e **`BoT-SORT`** usando **`vid_09`** como material principal de PoC com vagões abertos em movimento.
  3. **`vid_05`** restrito à análise visual e detecção pontual (não tracking contínuo por ser *timelapse*); **`vid_04`** avaliado separadamente para vagões estacionados/passarela; **`vid_07` excluído** do treinamento (vista lateral de contêineres, resolvendo a contradição anterior).
  4. **Fase 5 (material e carregamento) permanece pendente** até que novas fontes públicas abertas com superfície de carga visível sejam obtidas em quantidade e diversidade suficientes; a busca de novas fontes segue em paralelo.
- **Limites recomendados para upload na UI (Q15, baseados nos vídeos auditados):** manter `RAILVISION_UPLOAD_MAX_SIZE_MB=500` (cobre folgadamente vídeos Full HD/4K de 1–3 min como `vid_01`–`vid_09` entre 7 MB e 136 MB) e configurar limite sugerido de duração `<= 300s` na Fase 4/6 quando decodificação real entrar no backend.
