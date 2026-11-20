# Fase 3.1 — Proposta de Amostragem e Protocolo de Anotação (`open_top_freight_wagon`)

> [!IMPORTANT]
> **Status:** **Aprovado pelo usuário (`5 fps` / `61` frames + CVAT; extração de frames e pacotes CVAT concluída, aguardando *smoke test* e anotação manual).** Nenhum rótulo sintético ou treinamento foi iniciado.
>
> **Condições obrigatórias vigentes:**
> 1. Realizar primeiro o *smoke test* no CVAT (5 imgs detecção + 5 frames tracking), validando ordem dos frames, classe `0` e IDs (`specs/experiments/phase3_1-cvat-guide.md`).
> 2. Nunca usar anotações do tracking de teste (`vid_09`, `61` frames) no treinamento.
> 3. Revisar caixas e identidades (`qa_*.jpg`) antes de iniciar o treino ou calcular métricas.
> 4. Manter separação por campanhas (`vid_09` + `vid_16` juntas em `camp_vfli_cosne`) e preservar intactos os artefatos/checkpoints da Fase 3.
> 5. **Métricas MOT sobre 5 fps:** MOTA, IDF1, IDSW e fragmentações serão computadas estritamente sobre a **sequência amostrada a `5 fps` (`61` frames de `t=12,0s..24,0s`)**, e não sobre todos os frames nativos de `25 fps`. Os resultados permanecerão uma **validação inicial multi-campanha (4 campanhas)**, e não validação de generalização ampla (**Fase 5** segue pendente).

---

## 1. Inspeção Visual dos 5 Vídeos Públicos Selecionados

Todos os 5 vídeos locais em `data/raw/public-videos/` foram inspecionados via grades temporais densas (`vid_09`, `vid_14` completo em fatias de `45s`, `vid_15`, `vid_16` e `vid_04`) com seus metadados reais verificados por `OpenCV`:

| Vídeo | Campanha / Fonte independente | Dimensões · FPS · Duração (`frames`) | Autor · Licença | O que mostra e segmentos aproveitáveis / descartados |
|---|---|---|---|---|
| **`vid_14`** (`vid_14_coal_train_locomotive_exit_the_tunnel_cr.webm`) | **`camp_coal_tunnel`** *(Campanha 1 — EUA, carvão)* | `1280x720` · `59,94 fps` · `1.384,88s` (`83.010` frames) | RAILWAY BAY · `CC BY 3.0` | **Aproveitável:** `t=520..845s` (melhor trecho oblíquo elevado na encosta com gôndolas/hoppers de carvão em curva); `t=35..255s` (oblíquo à beira da estrada); `t=930..1110s` (oblíquo com oclusão parcial de pinheiros). **Negativos difíceis (*hard negatives*):** locomotivas BNSF/MRL próximas em `t=465..495s` e `t=1145..1165s`. **Descartados:** `t=0..25s` (preto), `t=280..385s` e `t=1190..1384s` (trem muito distante atrás de casas/vegetação). |
| **`vid_04`** (`vid_04_2es10_003_with_freight_train_from_above.webm`) | **`camp_perm_overpass`** *(Campanha 2 — Rússia, neve)* | `640x480` · `15,00 fps` · `56,73s` (`851` frames) | rustam79 · `CC BY 3.0` | **Aproveitável com parcimônia:** trem central em movimento tem locomotiva elétrica + vagões fechados com neve (**ótimo *hard negative***); vagões abertos verdes estão **estacionados** na via paralela direita parcialmente ocluídos por tanques cilíndricos e pelo próprio trem central. Como os vagões verdes são estáticos, apenas **10 frames** com mudanças reais de oclusão/iluminação evitam duplicatas quase idênticas. |
| **`vid_15`** (`vid_15_sishen_saldanha_railway_line.ogv`) | **`camp_sishen_saldanha`** *(Campanha 3 — África do Sul, minério de ferro)* | `1920x1080` · `29,97 fps` · `26,36s` (`790` frames) | Discott · `CC BY-SA 3.0` | **Aproveitável:** `t=0..18s` mostra passagem contínua de gôndolas vermelho-escuras altas em ângulo oblíquo elevado limpo; `t=19..24s` traz locomotivas de cauda vermelhas/amarelas (*hard negative* + últimas gôndolas). |
| **`vid_09`** (`vid_09_vossloh_g_1206_bb_of_vfli_video.webm`) | **`camp_vfli_cosne`** *(Campanha 4 — França, brita/lastro)* | `1920x1080` · `25,00 fps` · `49,80s` (`1.245` frames) | Cjp24 · `CC BY-SA 4.0` | **Aproveitável (principal teste móvel + tracking GT):** ~25 vagões *hopper* marrons carregados com brita cinza (`t=0..41s`) seguidos pela locomotiva vermelha/branca de cauda (`t=41..49,8s`). Possui vegetação na borda direita e postes de catenária cruzando os vagões (ótimo desafio realista de tracking). |
| **`vid_16`** (`vid_16_vossloh_g_1206_bb_61735_and_bb_61744_vfl.webm`) | **`camp_vfli_cosne`** *(MESMA campanha/local/autor de `vid_09`)* | `1920x1080` · `25,00 fps` · `20,40s` (`510` frames) | Cjp24 · `CC BY-SA 4.0` | **Mesma passarela e operadora de `vid_09`:** `t=0..11s` exibe duas locomotivas acopladas sem vagões visíveis (*hard negatives* puros); `t=13..20,4s` mostra o primeiro *hopper* aberto carregando brita nova. **Regra obrigatória:** deve permanecer **no mesmo split que `vid_09`**. |

---

## 2. Divisão Proposta por Campanha (Zero Vazamento) e Amostragem Justificada

Como existem apenas **4 campanhas visuais independentes** (agrupando `vid_09` + `vid_16` em `camp_vfli_cosne`), a amostragem não usa número arbitrário de frames, e sim **subamostragem proporcional à diversidade temporal real** de cada trecho útil:

| Split | Campanhas alocadas | Vídeos e estratégia de seleção de frames | Frames propostos | Caixas estimadas (`open_top_freight_wagon`) | Papel na Fase 3.1 |
|---|---|---|---|---|---|
| **`train`** *(2 campanhas)* | `camp_coal_tunnel` + `camp_perm_overpass` | - **`vid_14`:** `64` frames positivos espaçados por mudança de cena (`~5s` no trecho elevado `t=520..820s`, `~10s` em `t=35..255s` e `t=930..1110s`) + `8` frames negativos de locomotivas (`t=465..495s`, `t=1145..1165s`).<br>- **`vid_04`:** `10` frames em intervalos distintos (`t=2, 14, 24, 34, 42, 46, 50, 53, 55, 56,5s`) cobrindo antes, durante e depois da passagem do trem fechado central (evitando ~50 frames estáticos redundantes). | **`82`** (`72` `vid_14` + `10` `vid_04`) | **`~318`** (`~288` em `vid_14` + `~30` em `vid_04`) | *Fine-tuning* do `YOLO26n` em câmera oblíqua de carvão + negativos/neve, sem nunca ver `vid_15`, `vid_09` ou `vid_16`. |
| **`val`** *(1 campanha inédita)* | `camp_sishen_saldanha` | - **`vid_15`:** `19` frames espaçados a cada **`1,0s`** ao longo de `t=0..18s` (cobrindo vagões inteiros em movimento rápido) + `5` frames de transição da locomotiva de cauda (`t=19..23s`) para medir falsos positivos em locomotiva. | **`24`** | **`~98`** | Seleção de *checkpoint* e calibragem de confiança em **campanha inteiramente inédita** (`minério`, luz forte sul-africana). |
| **`test` (Detecção esparsa)** *(1 campanha inédita)* | `camp_vfli_cosne` (`vid_09` + `vid_16`) | - **`vid_09`:** `25` frames espaçados a cada **`~2,0s`** ao longo dos `49,8s` (inclui trecho dos *hoppers* e entrada da locomotiva traseira).<br>- **`vid_16`:** `4` frames só com as locomotivas acopladas (`t=2, 5, 8, 10s`, *hard negatives*) + `6` frames na entrada/passagem do primeiro *hopper* (`t=13,5..20,0s`). | **`35`** (`25` `vid_09` + `10` `vid_16`) | **`~106`** (`~100` `vid_09` + `~6` `vid_16`) | Medir ganho real de **detecção *cross-domain*** em vagões *hopper* abertos nunca vistos no treino/val. |
| **`test` (Tracking GT denso)** *(dentro de `camp_vfli_cosne`)* | `vid_09` *(trecho contínuo)* | - Trecho contínuo de **`12,0s`** (`t=12,0s..24,0s`, frames `300..600` a `25 fps`) amostrado a **`5 fps` (passo de 5 frames)** → passagem contínua e suave de ~`8–10` vagões *hopper* entrando à esquerda/fundo, atravessando o campo limpo e passando atrás do arbusto à direita.<br>*(Alternativa mais enxuta se preferir menor esforço: `2,5 fps` / passo de 10 frames = `31` frames, `~124` caixas).* | **`61`** frames sequenciais *(ou `31` @ `2,5 fps`)* | **`~244`** caixas com `track_id` persistente *(ou `~124` @ `2,5 fps`)* | Avaliar **`ByteTrack`** vs. **`BoT-SORT`** com métricas MOT reais (**MOTA**, **IDF1**, **ID Switches**, **Fragmentations**) sobre identidades anotadas. |

> [!NOTE]
> **Transparência sobre a limitação do split de treino:** Como `vid_09`+`vid_16` precisam ficar juntas no `test` e `vid_15` serve de `val` independente, `train` concentra a maior diversidade de movimento em `vid_14` complementada pelos `10` quadros em neve de `vid_04`. Portanto, mesmo após a Fase 3.1, o modelo treinado em 2 campanhas deverá ser tratado como **PoC especializada em `open_top_freight_wagon`**, devendo passar por expansão futura antes de qualquer uso geral.

---

## 3. Regra Única de Anotação (`0: open_top_freight_wagon`)

1. **Classe única (`0: open_top_freight_wagon`):** anotar o corpo completo externo de cada vagão aberto de carga a granel (gôndolas abertas e *hoppers* abertos de topo descoberto).
2. **Exclusões explícitas (NÃO anotar como classe `0`):** locomotivas, vagões-tanque cilíndricos, vagões fechados (`vid_04` trem central), contêineres, postes de catenária ou estruturas de ponte.
3. **Regra de corte na linha do horizonte / ponto de fuga:**
   - Anotar apenas vagões cujo retângulo visível tenha **lado menor `>= 24 px`** em `720p/1080p` (**`>= 18 px`** em `vid_04` `480p`), **`>= 40%` de área desobstruída** e **fronteira de acoplamento claramente separável** do vagão adjacente.
   - Vagões fundidos ao longe no ponto de fuga (onde o separador entre carros não é mais visível a olho nu) não devem ser chutados nem agrupados em uma única caixa de comboio.
4. **Identidade temporária no trecho de tracking (`vid_09`, `t=12..24s`):**
   - Cada vagão elegível pela regra de corte recebe um `track_id` inteiro crescente (`1, 2, 3, ...`) conforme entra na janela válida e o **mantém inalterado** enquanto atravessa a cena (mesmo quando parcialmente cruzado por fios de catenária ou pela borda do arbusto se `>= 40%` seguir visível).

---

## 4. Custo e Esforço Estimados de Anotação Manual

- **Ferramenta / licença de software:** **US\$ 0** utilizando ferramenta local *open-source* sem custo (ex.: **CVAT** local, **Label Studio**, ou um anotador leve local em HTML5/Canvas rodando direto no navegador sobre as imagens extraídas, salvando diretamente o `.txt` YOLO e MOT sem envio de imagens para nuvem externa).
- **Esforço manual estimado:**
  - **Pacote de detecção (`141` frames, `~522` caixas estimadas):** a `~6–10 s/caixa` + `~10 s` de revisão/negativos por frame = **`~1,3 a 1,9 horas`**.
  - **Clipe contínuo de tracking (`61` frames a `5 fps`, `~244` caixas com propagação/ajuste por frame anterior):** a `~3–5 s/caixa` = **`~0,5 a 0,8 horas`** *(ou `~0,3 a 0,5 horas` no modo reduzido `2,5 fps` / `31` frames)*.
  - **Revisão cruzada de consistência (corte de ponto de fuga e `track_id` contínuo):** **`~0,3 a 0,5 horas`**.
  - **Total estimado:** **`~2,1 a 3,2 horas`** no conjunto completo (**`~1,8 a 2,7 horas`** com tracking em `2,5 fps`).

---

## 5. Plano de Execução Pós-Aprovação (sem alterar escopo)

1. Extrair os pacotes de frames determinísticos + arquivo de manifesto/índice de campanha (`specs/data-manifests/open-top-split-v1.json`, fora do Git para as imagens brutas).
2. Executar a anotação manual humana conforme o fluxo escolhido pelo usuário (sem gerar *labels* automáticos ou sintéticos).
3. Treinar `YOLO26n` na GPU **`NVIDIA GeForce RTX 4060 Ti` (`cuda:0`)**, comparando ponto de partida `yolo26n.pt` (COCO) vs. `exp02/weights/best.pt` (TWIST) em `val` (`vid_15`), preservando todos os artefatos da Fase 3.
4. Avaliar detecção e tracking (`ByteTrack` vs. `BoT-SORT`) na campanha retida `camp_vfli_cosne` (`vid_09` + `vid_16`, `cuda:0`) contra as referências anotadas (MOTA, IDF1, IDSW, Det P/R @ IoU `0,50`, FPS efetivo e VRAM), registrando explicitamente dispositivo GPU e limitações antes de seguir ao **Spike Java/ONNX Runtime (Fase 4)**.
